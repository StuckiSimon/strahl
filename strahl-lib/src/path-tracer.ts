import { getBVHExtremes, MeshBVH } from "three-mesh-bvh";
import buildTracerShader from "./tracer-shader";
import buildRenderShader from "./render-shader";
import { logGroup } from "./benchmark/cpu-performance-logger.ts";
import { consolidateMesh } from "./consolidate-mesh";
import { OpenPBRMaterial } from "./openpbr-material";
import {
  getSizeAndAlignmentOfUnsizedArrayElement,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import { assertMeshBVHInternalStructure, bvhToTextures } from "./bvh-util";
import {
  CanvasReferenceError,
  InternalError,
  InvalidMaterialError,
  SignalAlreadyAbortedError,
  WebGPUNotSupportedError,
} from "./exceptions";
import {
  defaultEnvironmentLightConfig,
  EnvironmentLightConfig,
  getSunDirection,
} from "./environment-light";
import { isNil } from "./util/is-nil.ts";
import {
  CustomCameraSetup,
  isCustomCameraSetup,
  makeRawCameraSetup,
  ViewProjectionConfiguration,
} from "./camera";
import { buildAbortEventHub } from "./util/abort-event-hub.ts";

function prepareGeometry(model: any) {
  const reducedModel = consolidateMesh([model.scene]);
  const cpuLogGroup = logGroup("cpu");
  const boundsTree = new MeshBVH(reducedModel.geometry, {
    // This property is not officially supported by three-mesh-bvh just yet
    // @ts-ignore
    indirect: true,
  });

  const isStructureMatching = assertMeshBVHInternalStructure(boundsTree);
  if (!isStructureMatching) {
    throw new InternalError(
      "MeshBVH internal structure does not match, this indicates a change in the library which is not supported at prepareGeometry.",
    );
  }
  const extremes = getBVHExtremes(boundsTree);
  const correspondingExtremesEntry = extremes[0];
  const maxBvhDepth = correspondingExtremesEntry.depth.max;

  const { boundsArray, contentsArray } = bvhToTextures(boundsTree);
  const bvhBuildTime = cpuLogGroup.end();

  const meshPositions = boundsTree.geometry.attributes.position.array;
  const positions = meshPositions;

  const meshIndices = boundsTree.geometry.index!.array;

  const normals = boundsTree.geometry.attributes.normal.array;

  return {
    indirectBuffer: boundsTree._indirectBuffer,
    boundsArray,
    contentsArray,
    positions,
    normals,
    meshIndices,
    modelGroups: reducedModel.geometry.groups,
    modelMaterials: reducedModel.materials,
    maxBvhDepth,
    bvhBuildTime,
  };
}

type PathTracerOptions = {
  targetSamples?: number;
  kTextureWidth?: number;
  viewProjectionConfiguration?: ViewProjectionConfiguration;
  environmentLightConfiguration?: EnvironmentLightConfig;
  samplesPerIteration?: number;
  clearColor?: number[];
  maxRayDepth?: number;
  // todo: add real type
  finishedSampling?: (result: any) => void;
  signal?: AbortSignal;
};

async function runPathTracer(
  target: string,
  model: any,
  {
    targetSamples = 300,
    kTextureWidth = 512,
    viewProjectionConfiguration = {
      matrixWorldContent: [
        0.9348898557149565, 0, -0.354937963144642, 0, 0.04359232917084678,
        0.992429364980685, 0.1148201391807842, 0, 0.3522508573711748,
        -0.12281675587652569, 0.9278121458340784, 0, 63.44995297630283,
        -44.22427925573443, 209.99999999999994, 1,
      ],
      fov: 38.6701655,
      cameraTargetDistance: 200,
    },
    environmentLightConfiguration = defaultEnvironmentLightConfig(),
    samplesPerIteration = 1,
    clearColor = [1.0, 1.0, 1.0],
    maxRayDepth = 5,

    finishedSampling = () => {},
    signal = new AbortController().signal,
  }: PathTracerOptions = {},
) {
  /**
   * Map storing state of the tracer instance.
   * Routines with external resources should use this to be notified of aborts aka destruction of the path tracing process.
   */
  const abortEventHub = buildAbortEventHub();

  if (signal.aborted) {
    throw new SignalAlreadyAbortedError();
  }
  signal.addEventListener("abort", () => {
    abortEventHub.triggerAbort();
  });

  const TARGET_SAMPLES = targetSamples;
  const initLog = logGroup("init");

  const {
    indirectBuffer: indirectBufferData,
    boundsArray,
    contentsArray,
    positions,
    normals,
    maxBvhDepth,
    modelGroups,
    modelMaterials,
    meshIndices,
    bvhBuildTime,
  } = prepareGeometry(model);

  const canvas = document.getElementById(target);

  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new CanvasReferenceError();
  }

  if (!navigator.gpu) {
    throw new WebGPUNotSupportedError(
      "navigator.gpu is not available, most likely due to browser",
    );
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new WebGPUNotSupportedError("No suitable WebGPU adapter available");
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
  });

  abortEventHub.setDestructionNotifier("device", () => {
    device.destroy();
  });

  const timestampQuerySet = device.createQuerySet({
    type: "timestamp",
    count: 2,
  });

  const timestampQueryResolveBuffer = device.createBuffer({
    size: timestampQuerySet.count * 8,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE,
  });

  const timestampQueryResultBuffer = device.createBuffer({
    size: timestampQueryResolveBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new WebGPUNotSupportedError("No WebGPU context available");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
  });

  const kTextureHeight = kTextureWidth;

  const imageWidth = kTextureWidth;
  const imageHeight = imageWidth;

  const maxWorkgroupDimension = 16;

  const tracerShaderCode = buildTracerShader({
    imageWidth,
    imageHeight,
    maxWorkgroupDimension,
    maxBvhStackDepth: maxBvhDepth,
  });

  const textureData = new Uint8Array(kTextureWidth * kTextureHeight * 4);

  const texture = device.createTexture({
    size: [kTextureWidth, kTextureHeight],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.STORAGE_BINDING, // Permit writting to texture in compute shader
  });

  device.queue.writeTexture(
    { texture },
    textureData,
    { bytesPerRow: kTextureWidth * 4 },
    { width: kTextureWidth, height: kTextureHeight },
  );

  const textureB = device.createTexture({
    size: [kTextureWidth, kTextureHeight],
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.STORAGE_BINDING, // Permit writting to texture in compute shader
  });

  device.queue.writeTexture(
    {
      texture: textureB,
    },
    textureData,
    {
      bytesPerRow: kTextureWidth * 4,
    },
    {
      width: kTextureWidth,
      height: kTextureHeight,
    },
  );

  const sampler = device.createSampler();

  const renderShaderModule = device.createShaderModule({
    label: "Render Shader",
    code: buildRenderShader(),
  });

  const renderBindGroupLayout = device.createBindGroupLayout({
    label: "Texture sampler bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "filtering" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "float" },
      },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    label: "Pipeline Layout",
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Render pipeline",
    layout: renderPipelineLayout,
    vertex: {
      module: renderShaderModule,
      entryPoint: "vertexMain",
      buffers: [],
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "fragmentMain",
      targets: [
        {
          format,
        },
      ],
    },
  });

  const sceneMatrixWorld = model.scene.matrixWorld;

  // Prepare Position Data

  const isCustomCameraConfiguration = isCustomCameraSetup(
    viewProjectionConfiguration,
  );

  let cameraSetup: CustomCameraSetup;
  if (isCustomCameraConfiguration) {
    cameraSetup = viewProjectionConfiguration;
  } else {
    cameraSetup = makeRawCameraSetup(viewProjectionConfiguration, canvas);
  }

  const positionBuffer = device.createBuffer({
    label: "Position buffer",
    size: Float32Array.BYTES_PER_ELEMENT * positions.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const positionMapped = positionBuffer.getMappedRange();
  const positionData = new Float32Array(positionMapped);

  positionData.set(positions);
  positionBuffer.unmap();

  // Prepare Indices

  const indices = new Uint32Array(meshIndices);

  const indicesBuffer = device.createBuffer({
    label: "Index buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * indices.length,
    // todo: consider using GPUBufferUsage.INDEX
    usage: GPUBufferUsage.STORAGE, // GPUBufferUsage.INDEX,
    mappedAtCreation: true,
  });

  const indicesMapped = indicesBuffer.getMappedRange();
  const indicesData = new Uint32Array(indicesMapped);
  indicesData.set(indices);
  indicesBuffer.unmap();

  // Prepare Normal Data

  const normalBuffer = device.createBuffer({
    label: "Normal buffer",
    size: Float32Array.BYTES_PER_ELEMENT * normals.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const normalMapped = normalBuffer.getMappedRange();
  const normalData = new Float32Array(normalMapped);
  normalData.set(normals);
  normalBuffer.unmap();

  // Prepare BVH Bounds
  const boundsBuffer = device.createBuffer({
    label: "BVH bounds buffer",
    size: Float32Array.BYTES_PER_ELEMENT * boundsArray.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const boundsMapped = boundsBuffer.getMappedRange();
  const boundsData = new Float32Array(boundsMapped);
  boundsData.set(boundsArray);
  boundsBuffer.unmap();

  // Prepare BVH Contents
  const contentsBuffer = device.createBuffer({
    label: "BVH contents buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * contentsArray.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const contentsMapped = contentsBuffer.getMappedRange();
  const contentsData = new Uint32Array(contentsMapped);
  contentsData.set(contentsArray);
  contentsBuffer.unmap();

  // todo: reconsider type cast
  // Prepare BVH indirect buffer
  const indirectBuffer = device.createBuffer({
    label: "BVH indirect buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * indirectBufferData.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const indirectMapped = indirectBuffer.getMappedRange();
  const indirectData = new Uint32Array(indirectMapped);
  indirectData.set(indirectBufferData);
  indirectBuffer.unmap();

  // Prepare Object Definitions
  const OBJECT_DEFINITION_SIZE_PER_ENTRY = Uint32Array.BYTES_PER_ELEMENT * 3;
  const groups = modelGroups;

  const objectDefinitionsBuffer = device.createBuffer({
    label: "Object definitions buffer",
    size: OBJECT_DEFINITION_SIZE_PER_ENTRY * groups.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const objectDefinitionsMapped = objectDefinitionsBuffer.getMappedRange();
  const objectDefinitionsData = new Uint32Array(objectDefinitionsMapped);

  objectDefinitionsData.set(
    // todo: reconsider type assertion
    groups.map((g) => [g.start, g.count, g.materialIndex!]).flat(1),
  );
  objectDefinitionsBuffer.unmap();

  const materials = modelMaterials;

  // CODE#MEMORY-VIEW
  const definitions = makeShaderDataDefinitions(tracerShaderCode);
  const { size: bytesPerMaterial } = getSizeAndAlignmentOfUnsizedArrayElement(
    definitions.storages.materials,
  );

  const materialDataView = makeStructuredView(
    definitions.storages.materials,
    new ArrayBuffer(bytesPerMaterial * materials.length),
  );

  const materialBuffer = device.createBuffer({
    label: "Material buffer",
    size: bytesPerMaterial * materials.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    //mappedAtCreation: true,
  });

  // CODE#BUFFER-MAPPING
  materialDataView.set(
    materials.map((m) => {
      if (!(m instanceof OpenPBRMaterial)) {
        throw new InvalidMaterialError(m);
      }
      return {
        baseWeight: m.oBaseWeight,
        baseColor: m.oBaseColor,
        baseDiffuseRoughness: m.oBaseDiffuseRoughness,
        baseMetalness: m.oBaseMetalness,
        specularWeight: m.oSpecularWeight,
        specularColor: m.oSpecularColor,
        specularRoughness: m.oSpecularRoughness,
        specularAnisotropy: m.oSpecularRoughnessAnisotropy,
        specularIor: m.oSpecularIOR,
        coatWeight: m.oCoatWeight,
        coatColor: m.oCoatColor,
        coatRoughness: m.oCoatRoughness,
        coatRoughnessAnisotropy: m.oCoatRoughnessAnisotropy,
        // todo: Align casing for IOR parameter
        coatIor: m.oCoatIor,
        coatDarkening: m.oCoatDarkening,
        emissionLuminance: m.oEmissionLuminance,
        emissionColor: m.oEmissionColor,
        thinFilmThickness: m.oThinFilmThickness,
        thinFilmIOR: m.oThinFilmIOR,
      };
    }),
  );

  device.queue.writeBuffer(materialBuffer, 0, materialDataView.arrayBuffer);

  const computeBindGroupLayout = device.createBindGroupLayout({
    label: "Static compute bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 4,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const dynamicComputeBindGroupLayout = device.createBindGroupLayout({
    label: "Dynamic compute bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba8unorm" /*, access: "write-only"*/ },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba8unorm", access: "read-only" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  const computePipelineLayout = device.createPipelineLayout({
    label: "Compute pipeline layout",
    bindGroupLayouts: [computeBindGroupLayout, dynamicComputeBindGroupLayout],
  });

  const computePasses = Math.ceil(
    (imageWidth * imageWidth) / (maxWorkgroupDimension * maxWorkgroupDimension),
  );

  const computeShaderModule = device.createShaderModule({
    label: "Ray Tracing Compute Shader",
    code: tracerShaderCode,
  });

  const computeBindGroup = device.createBindGroup({
    label: "Static compute bind group",
    layout: computeBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: positionBuffer,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: indicesBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: boundsBuffer,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: contentsBuffer,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: normalBuffer,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: indirectBuffer,
        },
      },
      {
        binding: 6,
        resource: {
          buffer: objectDefinitionsBuffer,
        },
      },
      {
        binding: 7,
        resource: {
          buffer: materialBuffer,
        },
      },
    ],
  });

  const sunDirection = getSunDirection(environmentLightConfiguration.sun);

  initLog.end();

  const renderLoopStart = logGroup("render loop full");
  const buildRenderLoop = () => {
    let state: "running" | "halted" = "running";

    const isHalted = () => state === "halted";

    let currentAnimationFrameRequest: number | null = null;
    let currentSample = 0;
    let renderAgg = 0;
    let renderTimes = [];

    const render = async () => {
      const matrixWorld = cameraSetup.camera.matrixWorld;
      const invProjectionMatrix = cameraSetup.camera.projectionMatrixInverse;

      const renderLog = logGroup("render");
      const writeTexture = currentSample % 2 === 0 ? texture : textureB;
      const readTexture = currentSample % 2 === 0 ? textureB : texture;

      const { size: bytesForUniform } = definitions.uniforms.uniformData;

      const uniformData = makeStructuredView(
        definitions.uniforms.uniformData,
        new ArrayBuffer(bytesForUniform),
      );

      const uniformBuffer = device.createBuffer({
        label: "Uniform data buffer",
        size: bytesForUniform,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        // mappedAtCreation: true,
      });

      uniformData.set({
        invProjectionMatrix: invProjectionMatrix.elements,
        cameraWorldMatrix: matrixWorld.elements,
        invModelMatrix: sceneMatrixWorld.clone().invert().elements,
        seedOffset: Math.random() * Number.MAX_SAFE_INTEGER,
        priorSamples: currentSample,
        samplesPerPixel: samplesPerIteration,
        sunDirection,
        skyPower: environmentLightConfiguration.sky.power,
        skyColor: environmentLightConfiguration.sky.color,
        sunPower: Math.pow(10, environmentLightConfiguration.sun.power),
        sunAngularSize: environmentLightConfiguration.sun.angularSize,
        sunColor: environmentLightConfiguration.sun.color,
        clearColor: isNil(clearColor) ? [0, 0, 0] : clearColor,
        enableClearColor: isNil(clearColor) ? 0 : 1,
        maxRayDepth,
        objectDefinitionLength: groups.length,
      });
      // todo: consider buffer writing
      device.queue.writeBuffer(uniformBuffer, 0, uniformData.arrayBuffer);

      const dynamicComputeBindGroup = device.createBindGroup({
        label: "Dynamic compute bind group",
        layout: dynamicComputeBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: writeTexture.createView(),
          },
          {
            binding: 1,
            resource: readTexture.createView(),
          },
          {
            binding: 2,
            resource: {
              buffer: uniformBuffer,
            },
          },
        ],
      });

      const computePipeline = device.createComputePipeline({
        label: "Ray Tracing Compute pipeline",
        layout: computePipelineLayout,
        compute: {
          module: computeShaderModule,
          entryPoint: "computeMain",
        },
      });

      const encoder = device.createCommandEncoder();

      const computePass = encoder.beginComputePass({
        timestampWrites: {
          querySet: timestampQuerySet,
          beginningOfPassWriteIndex: 0,
          endOfPassWriteIndex: 1,
        },
      });
      computePass.setBindGroup(0, computeBindGroup);
      computePass.setBindGroup(1, dynamicComputeBindGroup);

      computePass.setPipeline(computePipeline);
      computePass.dispatchWorkgroups(
        Math.sqrt(computePasses),
        Math.sqrt(computePasses),
      );

      computePass.end();

      encoder.resolveQuerySet(
        timestampQuerySet,
        0,
        2,
        timestampQueryResolveBuffer,
        0,
      );

      if (timestampQueryResultBuffer.mapState === "unmapped") {
        encoder.copyBufferToBuffer(
          timestampQueryResolveBuffer,
          0,
          timestampQueryResultBuffer,
          0,
          timestampQueryResolveBuffer.size,
        );
      }

      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0.2, a: 1 },
            storeOp: "store",
          },
        ],
      });

      pass.setPipeline(renderPipeline);

      const renderBindGroup = device.createBindGroup({
        label: "Texture sampler bind group",
        layout: renderBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: sampler,
          },
          {
            binding: 1,
            resource: texture.createView(),
          },
        ],
      });

      pass.setBindGroup(0, renderBindGroup);
      const RENDER_TEXTURE_VERTEX_COUNT = 6;
      pass.draw(RENDER_TEXTURE_VERTEX_COUNT);

      pass.end();

      const commandBuffer = encoder.finish();

      device.queue.submit([commandBuffer]);

      if (timestampQueryResultBuffer.mapState === "unmapped") {
        try {
          await timestampQueryResultBuffer.mapAsync(GPUMapMode.READ);
        } catch (e) {
          // In case of planned cancellation, this is expected
          if (!abortEventHub.isRunning()) {
            console.warn("Aborted render loop");
            return;
          }
          throw e;
        }
        const data = new BigUint64Array(
          timestampQueryResultBuffer.getMappedRange(),
        );
        const gpuTime = data[1] - data[0];
        console.log(`GPU Time: ${gpuTime}ns`);
        timestampQueryResultBuffer.unmap();
      }

      const currentRenderTime = renderLog.end();
      renderTimes.push(currentRenderTime);
      renderAgg += currentRenderTime;

      if (currentSample < TARGET_SAMPLES && !isHalted()) {
        currentSample++;
        currentAnimationFrameRequest = requestAnimationFrame(render);
      } else {
        console.log("Average render time", renderAgg / TARGET_SAMPLES);
        currentAnimationFrameRequest = null;
        const fullRenderLoopTime = renderLoopStart.end();

        state = "halted";

        finishedSampling?.({
          bvhBuildTime,
          fullRenderLoopTime,
          allRenderTime: renderAgg,
          renderTimes: renderTimes,
        });
      }
    };
    currentAnimationFrameRequest = requestAnimationFrame(render);

    return {
      terminateLoop: () => {
        state = "halted";
        if (currentAnimationFrameRequest !== null) {
          cancelAnimationFrame(currentAnimationFrameRequest);
        }
      },
    };
  };

  let renderLoop = buildRenderLoop();

  const controlsChangeHandler = () => {
    renderLoop.terminateLoop();

    renderLoop = buildRenderLoop();
  };
  cameraSetup.controls?.addEventListener("change", controlsChangeHandler);
  abortEventHub.setDestructionNotifier("controls", () => {
    cameraSetup.controls?.removeEventListener("change", controlsChangeHandler);
  });
}

export default runPathTracer;

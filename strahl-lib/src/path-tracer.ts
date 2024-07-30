import { Matrix4, PerspectiveCamera, Vector3 } from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { MeshBVH } from "three-mesh-bvh";
import buildTracerShader from "./tracer-shader";
import buildRenderShader from "./render-shader";
import { logGroup } from "./cpu-performance-logger";
import { consolidateMesh } from "./consolidate-mesh";
import { OpenPBRMaterial } from "./openpbr-material";
import {
  getSizeAndAlignmentOfUnsizedArrayElement,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import { bvhToTextures } from "./bvh-util";

const sunConfig = {
  skyPower: 0.5,
  skyColor: [1.0, 1.0, 1.0],
  sunPower: 0.5,
  sunAngularSize: 40,
  sunLatitude: 45,
  sunLongitude: 180,
  sunColor: [1.0, 1.0, 0.9],
};

function getSunDirection() {
  const { sunLatitude, sunLongitude } = sunConfig;
  let latTheta = ((90.0 - sunLatitude) * Math.PI) / 180.0;
  let lonPhi = (sunLongitude * Math.PI) / 180.0;
  let cosTheta = Math.cos(latTheta);
  let sinTheta = Math.sin(latTheta);
  let cosPhi = Math.cos(lonPhi);
  let sinPhi = Math.sin(lonPhi);
  let x = sinTheta * cosPhi;
  let z = sinTheta * sinPhi;
  let y = cosTheta;
  return [x, y, z];
}

async function runPathTracer(target: string, model: any) {
  const initLog = logGroup("init");
  const canvas = document.getElementById("render-target");

  if (!(canvas instanceof HTMLCanvasElement)) {
    console.error("No canvas found");
    return;
  }

  if (!navigator.gpu) {
    console.error("WebGPU is not supported in your browser");
    return;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("No adapter found");
    return;
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
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
    console.error("No context found");
    return;
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
  });

  const kTextureWidth = 512;
  const kTextureHeight = kTextureWidth;

  const imageWidth = kTextureWidth;
  const imageHeight = imageWidth;

  const maxWorkgroupDimension = 16;

  const tracerShaderCode = buildTracerShader({
    imageWidth,
    imageHeight,
    maxWorkgroupDimension,
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

  const reducedModel = consolidateMesh([model.scene]);

  let matrixWorld = new Matrix4();

  // 45-cleaned
  matrixWorld.set(
    0.9136985580815294,
    0,
    -0.40639259953859086,
    0,
    0.06854721418549971,
    0.9856721677326017,
    0.15411572659764783,
    0,
    0.40056987453769,
    -0.16867239773393186,
    0.9006072383983735,
    0,
    40.14598145602767,
    -25.847614726242146,
    130.34925884569302,
    1,
  );

  matrixWorld.transpose();

  const camera = new PerspectiveCamera(38, 1, 0.01, 1000);
  const controls = new OrbitControls(camera, canvas);
  camera.matrixAutoUpdate = false;
  camera.applyMatrix4(matrixWorld);
  camera.matrixAutoUpdate = true;

  camera.updateMatrixWorld();

  const dir = new Vector3();
  camera.getWorldDirection(dir);
  const camTarget = camera.position.clone();
  camTarget.addScaledVector(dir, 330.39613);
  controls.target.copy(camTarget);

  controls.update();

  const boundsTree = new MeshBVH(reducedModel.geometry, {
    // This property is not officially supported by three-mesh-bvh just yet
    // @ts-ignore
    indirect: true,
  });

  const { boundsArray, contentsArray } = bvhToTextures(boundsTree);

  const meshPositions = boundsTree.geometry.attributes.position.array;

  const positions = meshPositions;

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
  const meshIndices = boundsTree.geometry.index!.array;

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
  const normals = boundsTree.geometry.attributes.normal.array;

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
    size:
      Uint32Array.BYTES_PER_ELEMENT *
      (boundsTree as unknown as { _indirectBuffer: ArrayLike<number> })
        ._indirectBuffer.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const indirectMapped = indirectBuffer.getMappedRange();
  const indirectData = new Uint32Array(indirectMapped);
  // todo: reconsider type cast
  indirectData.set(
    (boundsTree as unknown as { _indirectBuffer: ArrayLike<number> })
      ._indirectBuffer,
  );
  indirectBuffer.unmap();

  // Prepare Object Definitions
  const OBJECT_DEFINITION_SIZE_PER_ENTRY = Uint32Array.BYTES_PER_ELEMENT * 3;
  const groups = reducedModel.geometry.groups;

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

  const materials = reducedModel.materials;

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
  // todo: verify material type
  materialDataView.set(
    materials.map((m) => {
      if (!(m instanceof OpenPBRMaterial)) {
        console.error("Invalid material type", m);
        return;
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

  const sunDirection = getSunDirection();

  initLog.end();

  const TARGET_SAMPLES = 300;

  const renderLoopStart = logGroup("render loop full");
  const buildRenderLoop = () => {
    let state: "running" | "halted" = "running";

    const isHalted = () => state === "halted";

    let currentAnimationFrameRequest: number | null = null;
    let currentSample = 0;
    let renderAgg = 0;

    const render = async () => {
      const matrixWorld = camera.matrixWorld;
      const invProjectionMatrix = camera.projectionMatrixInverse;

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

      const SAMPLES_PER_ITERATION = 1;

      uniformData.set({
        invProjectionMatrix: invProjectionMatrix.elements,
        cameraWorldMatrix: matrixWorld.elements,
        invModelMatrix: sceneMatrixWorld.clone().invert().elements,
        seedOffset: Math.random() * Number.MAX_SAFE_INTEGER,
        priorSamples: currentSample,
        samplesPerPixel: SAMPLES_PER_ITERATION,
        sunDirection,
        skyPower: sunConfig.skyPower,
        skyColor: sunConfig.skyColor,
        sunPower: Math.pow(10, sunConfig.sunPower),
        sunAngularSize: sunConfig.sunAngularSize,
        sunColor: sunConfig.sunColor,
        clearColor: [1.0, 1.0, 1.0],
        enableClearColor: 1,
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
        await timestampQueryResultBuffer.mapAsync(GPUMapMode.READ);
        const data = new BigUint64Array(
          timestampQueryResultBuffer.getMappedRange(),
        );
        const gpuTime = data[1] - data[0];
        console.log(`GPU Time: ${gpuTime}ns`);
        timestampQueryResultBuffer.unmap();
      }

      renderAgg += renderLog.end();

      if (currentSample < TARGET_SAMPLES && !isHalted()) {
        currentSample++;
        currentAnimationFrameRequest = requestAnimationFrame(render);
      } else {
        console.log("Average render time", renderAgg / TARGET_SAMPLES);
        currentAnimationFrameRequest = null;
        renderLoopStart.end();

        state = "halted";
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

  controls.addEventListener("change", () => {
    renderLoop.terminateLoop();

    renderLoop = buildRenderLoop();
  });
}

export default runPathTracer;

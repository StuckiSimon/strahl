import buildTracerShader from "./tracer-shader";
import buildRenderShader from "./render-shader";
import buildDenoisePassShader from "./denoise-pass-shader.ts";
import buildTextureConverterPassShader from "./texture-converter-pass-shader.ts";
import { logGroup } from "./benchmark/cpu-performance-logger.ts";
import { OpenPBRMaterial } from "./openpbr-material";
import {
  getSizeAndAlignmentOfUnsizedArrayElement,
  makeShaderDataDefinitions,
  makeStructuredView,
} from "webgpu-utils";
import {
  CanvasReferenceError,
  InvalidMaterialError,
  SignalAlreadyAbortedError,
  WebGPUNotSupportedError,
} from "./core/exceptions.ts";
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
import { Group, Matrix4 } from "three";
import { prepareGeometry } from "./prepare-geometry.ts";
import { initUNetFromURL } from "oidn-web";

type GaussianConfig = {
  type: "gaussian";
  sigma?: number;
  kSigma?: number;
  threshold?: number;
};

type OIDNConfig = {
  type: "oidn";
  url?: string;
};

/**
 * Configuration options for the path tracer.
 */
export type PathTracerOptions = {
  targetSamples?: number;
  size?:
    | number
    | {
        width: number;
        height: number;
      };
  viewProjectionConfiguration?: ViewProjectionConfiguration;
  environmentLightConfiguration?: EnvironmentLightConfig;
  samplesPerIteration?: number;
  clearColor?: number[] | false;
  maxRayDepth?: number;
  finishedSampling?: (result: {
    bvhBuildTime: number;
    fullRenderLoopTime: number;
    allRenderTime: number;
    renderTimes: number[];
  }) => void;
  signal?: AbortSignal;
  enableTimestampQuery?: boolean;
  enableFloatTextureFiltering?: boolean;
  enableDenoise?: boolean | OIDNConfig | GaussianConfig;
};

async function denoise(
  {
    device,
    adapterInfo,
    url,
  }: { device: GPUDevice; adapterInfo: GPUAdapterInfo; url: string },
  {
    colorBuffer,
    albedoBuffer,
    normalBuffer,
  }: {
    colorBuffer: GPUBuffer;
    albedoBuffer: GPUBuffer;
    normalBuffer: GPUBuffer;
  },
  size: { width: number; height: number },
) {
  const unet = await initUNetFromURL(
    url,
    {
      device,
      adapterInfo,
    },
    {
      aux: true,
      hdr: true,
    },
  );

  type GPUImageData = {
    data: GPUBuffer;
    width: number;
    height: number;
  };

  return new Promise<GPUImageData>((resolve) => {
    unet.tileExecute({
      color: {
        data: colorBuffer,
        width: size.width,
        height: size.height,
      },
      albedo: {
        data: albedoBuffer,
        width: size.width,
        height: size.height,
      },
      normal: {
        data: normalBuffer,
        width: size.width,
        height: size.height,
      },
      done(finalBuffer) {
        resolve(finalBuffer);
      },
    });
  });
}

/**
 * Main routine to generate renderings.
 * @param target ID of the canvas element to render to
 * @param model The model to render
 * @param options Options for the path tracer
 * @returns A promise that resolves when the path tracer has finished setting up, but not when the path tracer has finished rendering.
 */
async function runPathTracer(
  target: string,
  model: { scene: Group },
  {
    targetSamples = 300,
    size = 512,
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
    enableTimestampQuery = true,
    enableFloatTextureFiltering = true,
    finishedSampling,
    signal = new AbortController().signal,
    enableDenoise = false,
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

  const supportsTimestampQuery = adapter.features.has("timestamp-query");
  const supportsFilterableFloatTexture =
    adapter.features.has("float32-filterable");

  const useTimestampQuery = enableTimestampQuery && supportsTimestampQuery;
  const useFloatTextureFiltering =
    enableFloatTextureFiltering && supportsFilterableFloatTexture;

  let featureList: GPUFeatureName[] = [];
  if (useTimestampQuery) {
    featureList.push("timestamp-query");
  }
  if (useFloatTextureFiltering) {
    featureList.push("float32-filterable");
  }

  const device = await adapter.requestDevice({
    requiredFeatures: featureList,
  });

  abortEventHub.setDestructionNotifier("device", () => {
    device.destroy();
  });

  let timestampQueryData: {
    querySet: GPUQuerySet;
    resolveBuffer: GPUBuffer;
    resultBuffer: GPUBuffer;
  } | null = null;
  if (useTimestampQuery) {
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

    timestampQueryData = {
      querySet: timestampQuerySet,
      resolveBuffer: timestampQueryResolveBuffer,
      resultBuffer: timestampQueryResultBuffer,
    };
  }

  const context = canvas.getContext("webgpu");

  if (!context) {
    throw new WebGPUNotSupportedError("No WebGPU context available");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
  });

  const width = typeof size === "number" ? size : size.width;
  const height = typeof size === "number" ? size : size.height;

  // denoise configuration
  const isOIDNConfig = (
    config: PathTracerOptions["enableDenoise"],
  ): config is OIDNConfig =>
    typeof config === "object" && config.type === "oidn";
  const isGaussianConfig = (
    config: PathTracerOptions["enableDenoise"],
  ): config is GaussianConfig =>
    typeof config === "object" && config.type === "gaussian";
  const denoiseMethod =
    (typeof enableDenoise === "object" && enableDenoise.type) ||
    (enableDenoise ? "gaussian" : "none");

  let gaussianConfig: Required<GaussianConfig> = {
    type: "gaussian",
    sigma: 4.0,
    kSigma: 1.0,
    threshold: 0.1,
  };
  let oidnConfig: Required<OIDNConfig> = {
    type: "oidn",
    url: "./oidn-weights/rt_hdr_alb_nrm.tza",
  };
  if (isGaussianConfig(enableDenoise)) {
    gaussianConfig = {
      ...gaussianConfig,
      ...enableDenoise,
    };
  } else if (isOIDNConfig(enableDenoise)) {
    oidnConfig = {
      ...oidnConfig,
      ...enableDenoise,
    };
  }

  const maxWorkgroupDimension = 16;

  const tracerShaderCode = buildTracerShader({
    maxBvhStackDepth: maxBvhDepth,
  });

  const textureData = new Float32Array(width * height * 4);

  const texture = device.createTexture({
    size: [width, height],
    format: "rgba32float",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.STORAGE_BINDING, // Permit writting to texture in compute shader
  });

  device.queue.writeTexture(
    { texture },
    textureData,
    { bytesPerRow: width * 4 * Float32Array.BYTES_PER_ELEMENT },
    { width: width, height: height },
  );

  const textureB = device.createTexture({
    size: [width, height],
    format: "rgba32float",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.STORAGE_BINDING, // Permit writting to texture in compute shader
  });

  device.queue.writeTexture(
    {
      texture: textureB,
    },
    textureData,
    {
      bytesPerRow: width * 4 * Float32Array.BYTES_PER_ELEMENT,
    },
    {
      width: width,
      height: height,
    },
  );

  const sampler = device.createSampler({
    magFilter: useFloatTextureFiltering ? "linear" : "nearest",
  });

  const renderShaderCode = buildRenderShader();

  const renderShaderModule = device.createShaderModule({
    label: "Render Shader",
    code: renderShaderCode,
  });

  const renderShaderDefinitions = makeShaderDataDefinitions(renderShaderCode);
  const { size: bytesForRenderUniform } =
    renderShaderDefinitions.uniforms.uniformData;
  const renderUniformData = makeStructuredView(
    renderShaderDefinitions.uniforms.uniformData,
    new ArrayBuffer(bytesForRenderUniform),
  );

  const renderUniformBuffer = device.createBuffer({
    label: "Render uniform data buffer",
    size: bytesForRenderUniform,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const setRenderUniformData = (enableDenoise: boolean) => {
    renderUniformData.set({
      textureWidth: width,
      textureHeight: height,
      denoiseSigma: gaussianConfig.sigma,
      denoiseKSigma: gaussianConfig.kSigma,
      denoiseThreshold: gaussianConfig.threshold,
      enableDenoise: enableDenoise ? 1 : 0,
    });
    device.queue.writeBuffer(
      renderUniformBuffer,
      0,
      renderUniformData.arrayBuffer,
    );
  };
  setRenderUniformData(false);

  const renderBindGroupLayout = device.createBindGroupLayout({
    label: "Texture sampler bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {
          type: useFloatTextureFiltering ? "filtering" : "non-filtering",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: useFloatTextureFiltering ? "float" : "unfilterable-float",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
        },
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

  const isCustomCameraConfiguration = isCustomCameraSetup(
    viewProjectionConfiguration,
  );

  let cameraSetup: CustomCameraSetup;
  if (isCustomCameraConfiguration) {
    cameraSetup = viewProjectionConfiguration;
  } else {
    cameraSetup = makeRawCameraSetup(viewProjectionConfiguration, canvas);
  }

  // Prepare Indices

  const indices = new Uint32Array(meshIndices);

  const indicesBuffer = device.createBuffer({
    label: "Index buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * indices.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const indicesMapped = indicesBuffer.getMappedRange();
  const indicesData = new Uint32Array(indicesMapped);
  indicesData.set(indices);
  indicesBuffer.unmap();

  // Prepare Geometry Data
  const sizePosition =
    Float32Array.BYTES_PER_ELEMENT * (positions.length / 3) * 4;
  const sizeNormal = Float32Array.BYTES_PER_ELEMENT * (normals.length / 3) * 4;

  const geometryBuffer = device.createBuffer({
    label: "Geometry buffer",
    size: sizePosition + sizeNormal,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const geometryMapped = geometryBuffer.getMappedRange();
  const geometryData = new Float32Array(geometryMapped);

  const itemsPerVertex = 3;

  for (let i = 0; i < positions.length; i += itemsPerVertex) {
    const offset = (i / itemsPerVertex) * 4 * 2;
    const offsetNormal = offset + 4;
    geometryData.set(positions.slice(i, i + itemsPerVertex), offset);
    geometryData.set(normals.slice(i, i + itemsPerVertex), offsetNormal);
  }

  geometryBuffer.unmap();

  // Prepare BVH Bounds
  const boundsBuffer = device.createBuffer({
    label: "BVH bounds buffer",
    size: Float32Array.BYTES_PER_ELEMENT * boundsArray.length,
    usage: GPUBufferUsage.STORAGE,
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
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const contentsMapped = contentsBuffer.getMappedRange();
  const contentsData = new Uint32Array(contentsMapped);
  contentsData.set(contentsArray);
  contentsBuffer.unmap();

  // Prepare BVH indirect buffer
  const indirectBuffer = device.createBuffer({
    label: "BVH indirect buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * indirectBufferData.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const indirectMapped = indirectBuffer.getMappedRange();
  const indirectData = new Uint32Array(indirectMapped);
  indirectData.set(indirectBufferData);
  indirectBuffer.unmap();

  // Prepare Object Definitions
  const OBJECT_DEFINITION_SIZE_PER_ENTRY = Uint32Array.BYTES_PER_ELEMENT * 3;

  const objectDefinitionsBuffer = device.createBuffer({
    label: "Object definitions buffer",
    size: OBJECT_DEFINITION_SIZE_PER_ENTRY * modelGroups.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const objectDefinitionsMapped = objectDefinitionsBuffer.getMappedRange();
  const objectDefinitionsData = new Uint32Array(objectDefinitionsMapped);

  objectDefinitionsData.set(
    modelGroups.map((g) => [g.start, g.count, g.materialIndex]).flat(1),
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
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
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

  const materialMapped = materialBuffer.getMappedRange();
  const materialMappedData = new Uint8Array(materialMapped);
  materialMappedData.set(new Uint8Array(materialDataView.arrayBuffer));
  materialBuffer.unmap();

  const computeBindGroupLayout = device.createBindGroupLayout({
    label: "Static compute bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 5,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 6,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        binding: 7,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
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
        storageTexture: { format: "rgba32float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba32float", access: "read-only" },
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
          buffer: geometryBuffer,
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

  const { size: bytesForUniform } = definitions.uniforms.uniformData;

  const uniformData = makeStructuredView(
    definitions.uniforms.uniformData,
    new ArrayBuffer(bytesForUniform),
  );

  const uniformBuffer = device.createBuffer({
    label: "Uniform data buffer",
    size: bytesForUniform,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const setUniformData = (
    invProjectionMatrix: Matrix4,
    matrixWorld: Matrix4,
    currentSample: number,
  ) => {
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
      clearColor: clearColor === false ? [0, 0, 0] : clearColor,
      enableClearColor: clearColor === false ? 0 : 1,
      maxRayDepth,
      objectDefinitionLength: modelGroups.length,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData.arrayBuffer);
  };

  initLog.end();

  const renderLoopStart = logGroup("render loop full");
  const buildRenderLoop = () => {
    let state: "running" | "halted" | "denoise" = "running";

    const isHalted = () => state === "halted";

    let currentAnimationFrameRequest: number | null = null;
    let currentSample = 0;
    let renderAgg = 0;
    let renderTimes = [];

    const render = async () => {
      const isLastSample = currentSample === TARGET_SAMPLES;
      if (isLastSample) {
        if (denoiseMethod === "gaussian") {
          setRenderUniformData(true);
        }
      }

      const matrixWorld = cameraSetup.camera.matrixWorld;
      const invProjectionMatrix = cameraSetup.camera.projectionMatrixInverse;

      const renderLog = logGroup("render");
      const writeTexture = currentSample % 2 === 0 ? texture : textureB;
      const readTexture = currentSample % 2 === 0 ? textureB : texture;

      setUniformData(invProjectionMatrix, matrixWorld, currentSample);

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
          constants: {
            wgSize: maxWorkgroupDimension,
            imageWidth: width,
            imageHeight: height,
          },
        },
      });

      const encoder = device.createCommandEncoder();

      const computePass = encoder.beginComputePass(
        isNil(timestampQueryData)
          ? undefined
          : {
              timestampWrites: {
                querySet: timestampQueryData.querySet,
                beginningOfPassWriteIndex: 0,
                endOfPassWriteIndex: 1,
              },
            },
      );
      computePass.setBindGroup(0, computeBindGroup);
      computePass.setBindGroup(1, dynamicComputeBindGroup);

      computePass.setPipeline(computePipeline);

      const dispatchX = Math.ceil(width / maxWorkgroupDimension);
      const dispatchY = Math.ceil(height / maxWorkgroupDimension);
      computePass.dispatchWorkgroups(dispatchX, dispatchY);

      computePass.end();

      if (!isNil(timestampQueryData)) {
        encoder.resolveQuerySet(
          timestampQueryData.querySet,
          0,
          2,
          timestampQueryData.resolveBuffer,
          0,
        );

        if (timestampQueryData.resultBuffer.mapState === "unmapped") {
          encoder.copyBufferToBuffer(
            timestampQueryData.resolveBuffer,
            0,
            timestampQueryData.resultBuffer,
            0,
            timestampQueryData.resolveBuffer.size,
          );
        }
      }

      const executeRenderPass = (
        texture: GPUTexture,
        encoder: GPUCommandEncoder,
      ) => {
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
            {
              binding: 2,
              resource: {
                buffer: renderUniformBuffer,
              },
            },
          ],
        });

        pass.setBindGroup(0, renderBindGroup);
        const RENDER_TEXTURE_VERTEX_COUNT = 6;
        pass.draw(RENDER_TEXTURE_VERTEX_COUNT);

        pass.end();

        const commandBuffer = encoder.finish();

        device.queue.submit([commandBuffer]);
      };

      executeRenderPass(texture, encoder);

      if (
        !isNil(timestampQueryData) &&
        timestampQueryData.resultBuffer.mapState === "unmapped"
      ) {
        try {
          await timestampQueryData.resultBuffer.mapAsync(GPUMapMode.READ);
        } catch (e) {
          // In case of planned cancellation, this is expected
          if (!abortEventHub.isRunning()) {
            console.warn("Aborted render loop");
            return;
          }
          throw e;
        }
        const data = new BigUint64Array(
          timestampQueryData.resultBuffer.getMappedRange(),
        );
        const gpuTime = data[1] - data[0];
        console.log(`GPU Time: ${gpuTime}ns`);
        timestampQueryData.resultBuffer.unmap();
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

        const activateDenoisePass = denoiseMethod === "oidn";

        state = "halted";

        if (activateDenoisePass) {
          state = "denoise";

          const dynamicComputeBindGroupLayout = device.createBindGroupLayout({
            label: "Dynamic denoise pass compute bind group layout",
            entries: [
              {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                  format: "rgba32float",
                },
              },
              {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: { format: "rgba32float", access: "read-only" },
              },
              {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type: "uniform",
                },
              },
              {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                  type: "storage",
                },
              },
            ],
          });

          const denoisePassShaderCode = buildDenoisePassShader({
            imageWidth: width,
            imageHeight: height,
            maxWorkgroupDimension,
            maxBvhStackDepth: maxBvhDepth,
          });

          const denoisePassDefinitions = makeShaderDataDefinitions(
            denoisePassShaderCode,
          );
          const { size: bytesForUniform } =
            denoisePassDefinitions.uniforms.uniformData;
          const uniformData = makeStructuredView(
            denoisePassDefinitions.uniforms.uniformData,
            new ArrayBuffer(bytesForUniform),
          );

          const buildDenoisePassUniformBuffer = (mode: 0 | 1) => {
            const uniformBuffer = device.createBuffer({
              label: "Denoise pass uniform data buffer",
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
              clearColor: clearColor === false ? [0, 0, 0] : clearColor,
              enableClearColor: clearColor === false ? 0 : 1,
              maxRayDepth,
              objectDefinitionLength: modelGroups.length,
              mode,
            });
            // todo: consider buffer writing
            device.queue.writeBuffer(uniformBuffer, 0, uniformData.arrayBuffer);

            return uniformBuffer;
          };

          const normalUniformBuffer = buildDenoisePassUniformBuffer(0);
          const albedoUniformBuffer = buildDenoisePassUniformBuffer(1);

          const computePipelineLayout = device.createPipelineLayout({
            label: "Dynamic denoise pass compute pipeline layout",
            bindGroupLayouts: [
              computeBindGroupLayout,
              dynamicComputeBindGroupLayout,
            ],
          });

          const float32ArrayImageSize = width * height * 4;

          const normalImageBuffer = device.createBuffer({
            label: "Normal image buffer",
            size: Float32Array.BYTES_PER_ELEMENT * float32ArrayImageSize,
            usage:
              GPUBufferUsage.STORAGE |
              // todo: check flags
              GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST,
            // mappedAtCreation: true,
          });

          const albedoImageBuffer = device.createBuffer({
            label: "Albedo image buffer",
            size: Float32Array.BYTES_PER_ELEMENT * float32ArrayImageSize,
            usage:
              GPUBufferUsage.STORAGE |
              // todo: check flags
              GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.COPY_DST,
            // mappedAtCreation: true,
          });

          const computeShaderModule = device.createShaderModule({
            label: "Denoise Pass Compute Shader",
            code: denoisePassShaderCode,
          });

          const computePipeline = device.createComputePipeline({
            label: "Denoise Pass Compute pipeline",
            layout: computePipelineLayout,
            compute: {
              module: computeShaderModule,
              entryPoint: "computeMain",
            },
          });

          const executeDenoisePass = (
            imageBuffer: GPUBuffer,
            uniformBuffer: GPUBuffer,
          ) => {
            const dynamicComputeBindGroup = device.createBindGroup({
              label: "Dynamic denoise pass compute bind group",
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
                {
                  binding: 3,
                  resource: {
                    buffer: imageBuffer,
                  },
                },
              ],
            });
            const encoder = device.createCommandEncoder();

            const computePass = encoder.beginComputePass();
            computePass.setBindGroup(0, computeBindGroup);
            computePass.setBindGroup(1, dynamicComputeBindGroup);

            computePass.setPipeline(computePipeline);

            const dispatchX = Math.ceil(width / maxWorkgroupDimension);
            const dispatchY = Math.ceil(height / maxWorkgroupDimension);
            computePass.dispatchWorkgroups(dispatchX, dispatchY);

            computePass.end();

            device.queue.submit([encoder.finish()]);
          };

          executeDenoisePass(normalImageBuffer, normalUniformBuffer);
          executeDenoisePass(albedoImageBuffer, albedoUniformBuffer);

          const textureBuffer = device.createBuffer({
            label: "Texture buffer",
            usage:
              GPUBufferUsage.COPY_DST |
              GPUBufferUsage.COPY_SRC |
              GPUBufferUsage.STORAGE,
            size: Float32Array.BYTES_PER_ELEMENT * 4 * width * height,
          });

          // fixme: use manual texture-converter-pass-shader
          {
            const textureConverterPassShaderCode =
              buildTextureConverterPassShader({
                imageWidth: width,
                imageHeight: height,
                maxWorkgroupDimension,
              });

            const computeShaderModule = device.createShaderModule({
              label: "Texture Converter Pass Compute Shader",
              code: textureConverterPassShaderCode,
            });

            const dynamicComputeBindGroupLayout = device.createBindGroupLayout({
              label: "Texture Converter pass compute bind group layout",
              entries: [
                {
                  binding: 0,
                  visibility: GPUShaderStage.COMPUTE,
                  storageTexture: {
                    format: "rgba32float",
                    access: "read-only",
                  },
                },
                {
                  binding: 1,
                  visibility: GPUShaderStage.COMPUTE,
                  buffer: {
                    type: "storage",
                  },
                },
              ],
            });

            const computePipelineLayout = device.createPipelineLayout({
              label: "Dynamic texture converter pass compute pipeline layout",
              bindGroupLayouts: [dynamicComputeBindGroupLayout],
            });

            const computePipeline = device.createComputePipeline({
              label: "Texture Converter Pass Compute pipeline",
              layout: computePipelineLayout,
              compute: {
                module: computeShaderModule,
                entryPoint: "computeMain",
              },
            });

            // todo: reconsider this pass as now the texture is already in the correct format
            const dynamicComputeBindGroup = device.createBindGroup({
              label: "Texture converter pass compute bind group",
              layout: dynamicComputeBindGroupLayout,
              entries: [
                {
                  binding: 0,
                  resource: writeTexture.createView(),
                },
                {
                  binding: 1,
                  resource: {
                    buffer: textureBuffer,
                  },
                },
              ],
            });

            const encoder = device.createCommandEncoder();

            const computePass = encoder.beginComputePass();
            computePass.setBindGroup(0, dynamicComputeBindGroup);

            computePass.setPipeline(computePipeline);

            const dispatchX = Math.ceil(width / maxWorkgroupDimension);
            const dispatchY = Math.ceil(height / maxWorkgroupDimension);
            computePass.dispatchWorkgroups(dispatchX, dispatchY);

            computePass.end();

            device.queue.submit([encoder.finish()]);
          }

          if (isHalted()) {
            return;
          }
          const outputBuffer = await denoise(
            { device, adapterInfo: adapter.info, url: oidnConfig.url },
            {
              colorBuffer: textureBuffer,
              albedoBuffer: albedoImageBuffer,
              normalBuffer: normalImageBuffer,
            },
            {
              width,
              height,
            },
          );

          if (isHalted()) {
            return;
          }

          {
            const textureFinal = device.createTexture({
              size: [width, height],
              format: "rgba32float",
              usage:
                GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.COPY_SRC,
            });
            const encoder = device.createCommandEncoder();
            encoder.copyBufferToTexture(
              // todo: this is used to debug the different buffers
              { buffer: outputBuffer.data, bytesPerRow: width * 4 * 4 },
              //{ buffer: normalImageBuffer, bytesPerRow: width * 4 * 4 },
              //{ buffer: textureBuffer, bytesPerRow: width * 4 * 4 },
              { texture: textureFinal },
              [width, height],
            );
            device.queue.submit([encoder.finish()]);

            const encoder2 = device.createCommandEncoder();
            executeRenderPass(textureFinal, encoder2);
          }

          textureBuffer.unmap();
        }

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

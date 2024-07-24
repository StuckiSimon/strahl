import { Matrix4, Mesh, PerspectiveCamera } from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
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

const DUCK_MODEL_URL = "models/duck/Duck.gltf";

const gltfLoader = new GLTFLoader();

async function loadGltf(url: string) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

// Build constants
const BYTES_PER_NODE = 6 * 4 + 4 + 4;

// See https://github.com/gkjohnson/three-mesh-bvh/blob/0eda7b718799e1709ad9efecdcc13c06ae3d5a55/src/core/utils/nodeBufferUtils.js
function isLeaf(n16: number, uint16Array: Uint16Array) {
  return uint16Array[n16 + 15] === 0xffff;
}

function getAtOffset(n32: number, uint32Array: Uint32Array) {
  return uint32Array[n32 + 6];
}

function getCount(n16: number, uint16Array: Uint16Array) {
  return uint16Array[n16 + 14];
}

function getRightNode(n32: number, uint32Array: Uint32Array) {
  return uint32Array[n32 + 6];
}

function getSplitAxis(n32: number, uint32Array: Uint32Array) {
  return uint32Array[n32 + 7];
}

function getBoundingDataIndex(n32: number) {
  return n32;
}

type MeshBVHInternal = {
  _roots: ArrayBuffer[];
};

const sunConfig = {
  skyPower: 0.4,
  skyColor: [0.5, 0.7, 1.0],
  sunPower: 0.9,
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

// CODE#BVH-TRANSFER
// Inspired by https://github.com/gkjohnson/three-mesh-bvh/blob/0eda7b718799e1709ad9efecdcc13c06ae3d5a55/src/gpu/MeshBVHUniformStruct.js#L110C1-L191C2
function bvhToTextures(bvh: MeshBVH) {
  const privateBvh = bvh as unknown as MeshBVHInternal;
  const roots = privateBvh._roots;

  if (roots.length !== 1) {
    throw new Error("MeshBVHUniformStruct: Multi-root BVHs not supported.");
  }

  const root = roots[0];
  const uint16Array = new Uint16Array(root);
  const uint32Array = new Uint32Array(root);
  const float32Array = new Float32Array(root);

  // Both bounds need two elements per node so compute the height so it's twice as long as
  // the width so we can expand the row by two and still have a square texture
  const nodeCount = root.byteLength / BYTES_PER_NODE;
  const boundsDimension = 2 * Math.ceil(Math.sqrt(nodeCount / 2));
  const boundsArray = new Float32Array(4 * boundsDimension * boundsDimension);

  const contentsDimension = Math.ceil(Math.sqrt(nodeCount));
  const contentsArray = new Uint32Array(
    2 * contentsDimension * contentsDimension,
  );

  for (let i = 0; i < nodeCount; i++) {
    const nodeIndex32 = (i * BYTES_PER_NODE) / 4;
    const nodeIndex16 = nodeIndex32 * 2;
    const boundsIndex = getBoundingDataIndex(nodeIndex32);
    for (let b = 0; b < 3; b++) {
      boundsArray[8 * i + 0 + b] = float32Array[boundsIndex + 0 + b];
      boundsArray[8 * i + 4 + b] = float32Array[boundsIndex + 3 + b];
    }

    if (isLeaf(nodeIndex16, uint16Array)) {
      const count = getCount(nodeIndex16, uint16Array);
      const offset = getAtOffset(nodeIndex32, uint32Array);

      const mergedLeafCount = 0xffff0000 | count;
      contentsArray[i * 2 + 0] = mergedLeafCount;
      contentsArray[i * 2 + 1] = offset;
    } else {
      const rightIndex =
        (4 * getRightNode(nodeIndex32, uint32Array)) / BYTES_PER_NODE;
      const splitAxis = getSplitAxis(nodeIndex32, uint32Array);

      contentsArray[i * 2 + 0] = splitAxis;
      contentsArray[i * 2 + 1] = rightIndex;
    }
  }

  return {
    boundsArray,
    contentsArray,
    boundsDimension,
    contentsDimension,
  };
}

async function run() {
  const duck = await loadGltf(DUCK_MODEL_URL);

  const duckMesh = duck.scene.children[0].children[0] as Mesh;

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
  const model = duckMesh;

  const sceneMatrixWorld = model.scene.matrixWorld;

  // Prepare Position Data

  // todo: remove magic downscaling & transform
  const scaleFactor = 0.01;
  const scaleMatrix = new Matrix4().makeScale(
    scaleFactor,
    scaleFactor,
    scaleFactor,
  );
  duckMesh.geometry.applyMatrix4(scaleMatrix);

  const translateX = 0;
  const translateY = -1;
  const translateZ = -2;

  const transformMatrix = new Matrix4().makeTranslation(
    translateX,
    translateY,
    translateZ,
  );

  duckMesh.geometry.applyMatrix4(transformMatrix);
  const reducedModel = consolidateMesh([duckMesh]);

  const camera = new PerspectiveCamera();
  camera.position.set(0.0, 0.0, 30.0);
  camera.lookAt(0, 2, 0);

  camera.updateMatrixWorld(true);

  reducedModel.geometry.applyMatrix4(transformMatrix);
  reducedModel.boundsTree = new MeshBVH(reducedModel.geometry, {
    indirect: true,
  });
  const boundsTree = reducedModel.boundsTree;

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

  // Prepare BVH indirect buffer
  const indirectBuffer = device.createBuffer({
    label: "BVH indirect buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * boundsTree._indirectBuffer.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const indirectMapped = indirectBuffer.getMappedRange();
  const indirectData = new Uint32Array(indirectMapped);
  indirectData.set(boundsTree._indirectBuffer);
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
    groups.map((g) => [g.start, g.count, g.materialIndex]).flat(1),
  );
  objectDefinitionsBuffer.unmap();

  const materials = reducedModel.materials;

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

  const TARGET_SAMPLES = 20;

  const controls = new OrbitControls(camera, canvas);

  const buildRenderLoop = () => {
    let state: "running" | "halted" = "running";

    const isHalted = () => state === "halted";

    let currentAnimationFrameRequest: number | null = null;
    let currentSample = 0;
    let renderAgg = 0;

    const render = async () => {
      const projectionMatrix = camera.projectionMatrix;
      const matrixWorld = camera.matrixWorld;
      const invProjectionMatrix = projectionMatrix.clone().invert();

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

run();

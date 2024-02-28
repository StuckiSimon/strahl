import buildTracerShader from "./tracer-shader";

function logGroup(label: string) {
  const start = window.performance.now();

  return {
    end() {
      const end = window.performance.now();
      console.log(`${label}: ${end - start}ms`);
    },
  };
}

async function run() {
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

  const device = await adapter.requestDevice();

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

  const kTextureWidth = 1024;
  const kTextureHeight = kTextureWidth;

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
    { width: kTextureWidth, height: kTextureHeight }
  );

  const sampler = device.createSampler();

  const renderShaderModule = device.createShaderModule({
    label: "Cell shader",
    code: `
const pos = array(
    // 1
    vec2f(-1.0, -1.0),
    vec2f(1.0, -1.0),
    vec2f(-1.0, 1.0),

    // 2
    vec2f( -1.0,  1.0),
    vec2f( 1.0,  -1.0),
    vec2f( 1.0,  1.0),
  );

struct VertexInput {
  @builtin(vertex_index) instance: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) texcoord: vec2f,
};

fn convertToZeroOne(position: vec2f) -> vec2f {
  return (position + 1.0) / 2;
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let xy = pos[input.instance];
  output.pos = vec4f(xy, 0, 1);

  let baseCoord = convertToZeroOne(vec2f(xy.x, xy.y));
  output.texcoord = vec2f(baseCoord.x, 1.0 - baseCoord.y);
  return output;
}

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
    return textureSample(texture, texture_sampler, input.texcoord);
}
    `,
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

  type Vec3 = [number, number, number];

  type Sphere = {
    center: Vec3;
    radius: number;
  };

  const spheres: Sphere[] = [
    {
      center: [0, 0, -1],
      radius: 0.5,
    },
    {
      center: [0, -100.5, -1],
      radius: 100,
    },
    {
      center: [1, 0, -1],
      radius: 0.5,
    },
  ];

  type Aabb = {
    min: Vec3;
    max: Vec3;
  };

  type BvhNode = {
    boundingBox: Aabb;
    leftIndex: number;
    rightIndex: number;
    sphereIndex: number;
  };

  function getBoundingBoxFromSphere(sphere: Sphere): Aabb {
    const [x, y, z] = sphere.center;
    const r = sphere.radius;
    return {
      min: [x - r, y - r, z - r],
      max: [x + r, y + r, z + r],
    };
  }

  function combineBoundingBoxes(a: Aabb, b: Aabb): Aabb {
    return {
      min: [
        Math.min(a.min[0], b.min[0]),
        Math.min(a.min[1], b.min[1]),
        Math.min(a.min[2], b.min[2]),
      ],
      max: [
        Math.max(a.max[0], b.max[0]),
        Math.max(a.max[1], b.max[1]),
        Math.max(a.max[2], b.max[2]),
      ],
    };
  }

  const bvs: BvhNode[] = [
    {
      boundingBox: {
        // x, y, z
        min: [-100, -100, -100],
        max: [100, 100, 100],
      },
      leftIndex: 1,
      rightIndex: 2,
      sphereIndex: -1,
    },
    {
      boundingBox: combineBoundingBoxes(
        getBoundingBoxFromSphere(spheres[0]),
        getBoundingBoxFromSphere(spheres[2])
      ),
      leftIndex: 3,
      rightIndex: 4,
      sphereIndex: -1,
    },
    {
      boundingBox: getBoundingBoxFromSphere(spheres[1]),
      leftIndex: -1,
      rightIndex: -1,
      sphereIndex: 1,
    },
    {
      boundingBox: getBoundingBoxFromSphere(spheres[0]),
      leftIndex: -1,
      rightIndex: -1,
      sphereIndex: 0,
    },
    {
      boundingBox: getBoundingBoxFromSphere(spheres[2]),
      leftIndex: -1,
      rightIndex: -1,
      sphereIndex: 2,
    },
  ];

  const FLOATS_PER_BVH_NODE = 9;
  const aabbBuffer = device.createBuffer({
    label: "AABB buffer",
    size: FLOATS_PER_BVH_NODE * Float32Array.BYTES_PER_ELEMENT * bvs.length,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  const aabbMapped = aabbBuffer.getMappedRange();
  const aabbFloatData = new Float32Array(aabbMapped);
  const aabbIntegerData = new Uint32Array(aabbMapped);

  const indexNodes = bvs.map((node, i) => ({ ...node, index: i }));
  for (const node of indexNodes) {
    const { index, boundingBox, leftIndex, rightIndex, sphereIndex } = node;
    const { min, max } = boundingBox;

    const offset = index * FLOATS_PER_BVH_NODE;

    aabbFloatData[offset] = min[0];
    aabbFloatData[offset + 1] = max[0];
    aabbFloatData[offset + 2] = min[1];
    aabbFloatData[offset + 3] = max[1];
    aabbFloatData[offset + 4] = min[2];
    aabbFloatData[offset + 5] = max[2];

    //aabbFloatData.set(min, offset);
    //aabbFloatData.set(max, offset + 3);
    aabbIntegerData[offset + 6] = leftIndex;
    aabbIntegerData[offset + 7] = rightIndex;
    aabbIntegerData[offset + 8] = sphereIndex;
  }

  aabbBuffer.unmap();

  const computeBindGroupLayout = device.createBindGroupLayout({
    label: "Compute bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba8unorm" /*, access: "write-only"*/ },
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
    label: "Compute pipeline layout",
    bindGroupLayouts: [computeBindGroupLayout],
  });

  const computeBindGroup = device.createBindGroup({
    label: "Compute bind group",
    layout: computeBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: texture.createView(),
      },
      {
        binding: 1,
        resource: {
          buffer: aabbBuffer,
        },
      },
    ],
  });

  const imageWidth = kTextureWidth;
  const imageHeight = imageWidth;

  const maxWorkgroupDimension = 16;
  const computePasses = Math.ceil(
    // todo: check…
    (imageWidth * imageWidth) / maxWorkgroupDimension
  );

  const computeShaderModule = device.createShaderModule({
    label: "Ray Tracing Compute Shader",
    code: buildTracerShader({
      imageWidth,
      imageHeight,
      maxWorkgroupDimension,
    }),
  });

  const computePipeline = device.createComputePipeline({
    label: "Ray Tracing Compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: "computeMain",
    },
  });

  initLog.end();

  const computeLog = logGroup("compute");

  let TARGET_FRAMES = 0;
  let frame = 0;
  const render = () => {
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setBindGroup(0, computeBindGroup);

    computePass.setPipeline(computePipeline);
    computePass.dispatchWorkgroups(
      Math.sqrt(computePasses),
      Math.sqrt(computePasses)
    );

    computePass.end();

    computeLog.end();
    const renderLog = logGroup("render");

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

    pass.setBindGroup(0, renderBindGroup);
    const RENDER_TEXTURE_VERTEX_COUNT = 6;
    pass.draw(RENDER_TEXTURE_VERTEX_COUNT);

    pass.end();

    const commandBuffer = encoder.finish();
    renderLog.end();

    const queueSubmitLog = logGroup("queue submit");
    device.queue.submit([commandBuffer]);
    queueSubmitLog.end();

    if (frame < TARGET_FRAMES) {
      frame++;
      requestAnimationFrame(render);
    }
  };
  requestAnimationFrame(render);
}

run();

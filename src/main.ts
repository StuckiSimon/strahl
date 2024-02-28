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
    code: `
@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> bvhNodes: array<BvhNode>;

const focal_length = 1.0;
const viewport_height = 4.0;
const viewport_width = 4.0;
const camera_center = vec3<f32>(0.0, 0.0, 0.0);

const viewport_u = vec3<f32>(viewport_width, 0.0, 0.0);
const viewport_v = vec3<f32>(0.0, -viewport_height, 0.0);

const pixel_delta_u = viewport_u / ${imageWidth};
const pixel_delta_v = viewport_v / ${imageHeight};

const viewport_upper_left = camera_center - vec3<f32>(0, 0, focal_length) - viewport_u / 2.0 - viewport_v / 2.0;
const pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

const samples_per_pixel = 30;
const max_depth = 5;

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
};

struct MaterialDefinition {
  materialType: i32,
  index: i32,
}

struct HitRecord {
  point: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  front_face: bool,
  material: MaterialDefinition,
}

alias Color = vec3<f32>;

struct LambertianMaterial {
  texture: Color,
}

const LAMBERTIAN_MATERIAL_TYPE = 0;

struct DiffuseLightMaterial {
  emit: Color,
}

const DIFFUSE_LIGHT_MATERIAL_TYPE = 1;

struct Triangle {
  Q: vec3<f32>,
  u: vec3<f32>,
  v: vec3<f32>,
  material: MaterialDefinition,
}

struct Interval {
  min: f32,
  max: f32,
}

fn interval(min: f32, max: f32) -> Interval {
  return Interval(min, max);
}

fn interval_contains(interval: Interval, value: f32) -> bool {
  return interval.min <= value && value <= interval.max;
}

fn interval_surrounds(interval: Interval, value: f32) -> bool {
  return interval.min < value && value < interval.max;
}

fn interval_clamp(interval: Interval, value: f32) -> f32 {
  return clamp(value, interval.min, interval.max);
}

struct Aabb {
  interval_x: Interval,
  interval_y: Interval,
  interval_z: Interval,
}

fn aabb_axis(aabb: Aabb, axis: i32) -> Interval {
  if (axis == 0) {
    return aabb.interval_x;
  } else if (axis == 1) {
    return aabb.interval_y;
  } else {
    return aabb.interval_z;
  }
}

fn aabb_hit(aabb: Aabb, r: Ray, ray_t: ptr<function, Interval>) -> bool {
  for (var a = 0; a < 3; a += 1) {
    let t0 = min((aabb_axis(aabb, a).min - r.origin[a]) / r.direction[a],
                 (aabb_axis(aabb, a).max - r.origin[a]) / r.direction[a]);
    let t1 = max((aabb_axis(aabb, a).min - r.origin[a]) / r.direction[a],
                  (aabb_axis(aabb, a).max - r.origin[a]) / r.direction[a]);
    (*ray_t).min = max((*ray_t).min, t0);
    (*ray_t).max = min((*ray_t).max, t1);
    if ((*ray_t).max <= (*ray_t).min) {
      return false;
    }
  }
  return true;
}

struct BvhNode {
  boundingBox: Aabb,
  leftIndex: i32,
  rightIndex: i32,
  sphereIndex: i32,
}

fn nearZero(v: vec3f) -> bool {
  let epsilon = vec3f(1e-8);
  return any(abs(v) < epsilon);
}

fn render_lambertian_material(material: LambertianMaterial, hit_record: HitRecord, attenuation: ptr<function, Color>, emission_color: ptr<function, Color>, scattered: ptr<function, Ray>, seed: ptr<function, u32>) -> bool {
  (*emission_color) = Color(0.0,0.0,0.0);
  
  var scatter_direction = hit_record.normal + normalize(randInUnitSphere(seed));

  // Catch degenerate scatter direction
  if (nearZero(scatter_direction)) {
    scatter_direction = hit_record.normal;
  }

  (*scattered) = Ray(hit_record.point, scatter_direction);

  (*attenuation) = material.texture;

  return true;  
}

fn render_diffuse_light_material(material: DiffuseLightMaterial, hit_record: HitRecord, attenuation: ptr<function, Color>, emission_color: ptr<function, Color>, scattered: ptr<function, Ray>, seed: ptr<function, u32>) -> bool {
  (*emission_color) = material.emit;
  return false;
}


const lambertianMaterials: array<LambertianMaterial, 2> = array<LambertianMaterial, 2>(
  LambertianMaterial(vec3<f32>(0.5, 1.0, 0.0)),
  LambertianMaterial(vec3<f32>(1.0, 0.5, 0.2)),
);

const diffuseLightMaterials: array<DiffuseLightMaterial, 2> = array<DiffuseLightMaterial, 2>(
  DiffuseLightMaterial(vec3<f32>(1.0, 0.5, 1.0)),
  DiffuseLightMaterial(vec3<f32>(1.0, 1.0, 1.0)),
);

const defaultMaterial = MaterialDefinition(LAMBERTIAN_MATERIAL_TYPE, 0);
const defaultDiffuseLightMaterial = MaterialDefinition(DIFFUSE_LIGHT_MATERIAL_TYPE, 0);

const TRIANGLE_COUNT = 6;
// Triangles are encoded as first being the lower point, then the two edges
const triangles: array<Triangle, TRIANGLE_COUNT> = array<Triangle, TRIANGLE_COUNT>(
  // wall facing camera
  Triangle(vec3<f32>(-3, 0, -3), vec3<f32>(4, 0, 0), vec3<f32>(0, 4, 0), MaterialDefinition(LAMBERTIAN_MATERIAL_TYPE, 1)),
  Triangle(vec3<f32>(-3, 4, -3), vec3<f32>(4, -4, 0), vec3<f32>(4, 0, 0), MaterialDefinition(LAMBERTIAN_MATERIAL_TYPE, 1)),
  // ground floor below the wall
  Triangle(vec3<f32>(-3, 0, -3), vec3<f32>(4, 0, 0), vec3<f32>(0, -2, 1.5), defaultMaterial),
  Triangle(vec3<f32>(1, 0, -3), vec3<f32>(0, -2, 1.5), vec3<f32>(-4, -2, 1.5), defaultMaterial),
  // Light source from the right
  Triangle(vec3<f32>(2, 5, -6), vec3<f32>(0.1, -20, 0), vec3<f32>(0, 0, 10), defaultDiffuseLightMaterial),
  // Light from the left
  Triangle(vec3<f32>(-3, 5, -6), vec3<f32>(0.0, -20, 0), vec3<f32>(0, 0, 10), MaterialDefinition(DIFFUSE_LIGHT_MATERIAL_TYPE, 1)),
);

fn bvh_node_hit(bvh: BvhNode, r: Ray, ray_t: ptr<function, Interval>) -> bool {
  return aabb_hit(bvh.boundingBox, r, ray_t);
}

fn ray_at(ray: Ray, t: f32) -> vec3<f32> {
  return ray.origin + t * ray.direction;
}

fn length_squared(v: vec3<f32>) -> f32 {
  return dot(v, v);
}

// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L283
const PCG_INC = 2891336453u;
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L278
const PCG_MULTIPLIER = 747796405u;

// https://www.pcg-random.org/download.html#id1
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L1533
fn randInt(seed: ptr<function, u32>) -> i32 {
  let oldstate = *seed;
  *seed = *seed * PCG_MULTIPLIER + PCG_INC;
  let word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
  return i32((word >> 22u) ^ word);
}

fn randFloat(seed: ptr<function, u32>) -> f32 {
  let val = randInt(seed);
  return f32(val) / f32(0xffffffffu);
}

fn randFloatInRange(min: f32, max: f32, seed: ptr<function, u32>) -> f32 {
  return (randFloat(seed) * (max - min)) + min;
}

fn randVec3(seed: ptr<function, u32>) -> vec3<f32> {
  return vec3<f32>(randFloat(seed), randFloat(seed), randFloat(seed));
}

fn randVec3InRange(min: f32, max: f32, seed: ptr<function, u32>) -> vec3<f32> {
  return vec3<f32>(
    randFloatInRange(min, max, seed),
    randFloatInRange(min, max, seed),
    randFloatInRange(min, max, seed),
  );
}

fn randInUnitSphere(seed: ptr<function, u32>) -> vec3<f32> {
  while (true) {
    let p = randVec3(seed);
    if (length_squared(p) >= 1.0) {
      continue;
    }
    return p;
  }
  // Should never reach here
  return vec3<f32>(0.0, 0.0, 0.0);
}

fn randomOnHemiSphere(normal: vec3<f32>, seed: ptr<function, u32>) -> vec3<f32> {
  let in_unit_sphere = randInUnitSphere(seed);
  return select(in_unit_sphere, -in_unit_sphere, dot(in_unit_sphere, normal) < 0);
}

fn triangle_bounding_box(triangle: Triangle) -> Aabb {
  let min = min(triangle.Q, min(triangle.Q + triangle.u, triangle.Q + triangle.v));
  let max = max(triangle.Q, max(triangle.Q + triangle.u, triangle.Q + triangle.v));
  return Aabb(interval(min.x, max.x), interval(min.y, max.y), interval(min.z, max.z));
}

fn triangle_hit(triangle: Triangle, ray: Ray, ray_t: Interval, hit_record: ptr<function, HitRecord>) -> bool {
  let edge1 = triangle.u;
  let edge2 = triangle.v;
  let h = cross(ray.direction, edge2);
  let a = dot(edge1, h);
  // No hit if ray is parallel to the triangle
  if (a > -0.00001 && a < 0.00001) {
    return false;
  }
  let f = 1.0 / a;
  let s = ray.origin - triangle.Q;
  let u = f * dot(s, h);
  // No hit if ray is outside the triangle
  if (u < 0.0 || u > 1.0) {
    return false;
  }
  let q = cross(s, edge1);
  let v = f * dot(ray.direction, q);
  // No hit if ray is outside the triangle
  if (v < 0.0 || u + v > 1.0) {
    return false;
  }
  let t = f * dot(edge2, q);
  // No hit if triangle is behind the ray
  if (t < (ray_t).min || t > (ray_t).max) {
    return false;
  }

  (*hit_record).t = t;
  (*hit_record).point = ray_at(ray, t);
  (*hit_record).normal = normalize(cross(edge1, edge2));
  return true;
}

fn hittable_list_hit(ray: Ray, ray_t: Interval, hit_record: ptr<function, HitRecord>) -> bool {
  var temp_record: HitRecord;
  var hit_anything = false;
  var closest_so_far = ray_t.max;

  for (var i = 0; i < TRIANGLE_COUNT; i += 1) {
    let triangle = triangles[i];
    if (triangle_hit(triangle, ray, Interval(ray_t.min, closest_so_far), &temp_record)) {
      hit_anything = true;
      closest_so_far = temp_record.t;
      (*hit_record) = temp_record;
      // FIXME: Add this also to other hit checks!
      (*hit_record).material = triangle.material;
    }
  }

  return hit_anything;
}

struct BouncingInfo {
  attenuation: Color,
  emission: Color,
}

fn ray_color(ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
  var hit_record: HitRecord;
  var local_ray = ray;

  var color_stack: array<BouncingInfo, max_depth>;
  var color_stack_idx = 0;

  for (var i = 0; i < max_depth; i += 1) {
    if (hittable_list_hit(local_ray, Interval(0.001, 999999999999999999), &hit_record)) {
      var attenuation: Color;
      
      //color = 0.5 * color;
      var emission_color = Color(0,0,0);
      
      let material = hit_record.material;
      var scattered = false;
      if (material.materialType == LAMBERTIAN_MATERIAL_TYPE) {
        scattered = render_lambertian_material(
        lambertianMaterials[material.index], hit_record, &attenuation, &emission_color, &local_ray, seed);
        } else if (material.materialType == DIFFUSE_LIGHT_MATERIAL_TYPE) {
        scattered = render_diffuse_light_material(
        diffuseLightMaterials[material.index], hit_record, &attenuation, &emission_color, &local_ray, seed);
      }

      if (!scattered) {
        color_stack[color_stack_idx] = BouncingInfo(Color(0.0,0.0,0.0), emission_color);
        break;
      } else {
        color_stack[color_stack_idx] = BouncingInfo(attenuation, Color(0.0,0,0)); //BouncingInfo(attenuation, emission_color);
      }
    } else {      
      let unit_direction = normalize(local_ray.direction);
      let a = 0.5 * (unit_direction.y + 1.0);
      color_stack_idx -= 1;
      break;
    }
    color_stack_idx += 1;
  }

  var color = Color(0,0,0); //color_stack[color_stack_idx].emission;
  let last_idx = color_stack_idx;
  for (var i = color_stack_idx; i >= 0; i -= 1) {
    let bouncing = color_stack[i];
    color = bouncing.emission + (bouncing.attenuation * color);
  }

  return color;
}

fn get_ray(i: f32, j: f32, seed: ptr<function, u32>) -> Ray {
  let pixel_center = pixel00_loc + (i * pixel_delta_u) + (j * pixel_delta_v);
  let pixel_sample = pixel_center + pixel_sample_square(seed);

  let ray_origin = camera_center;
  let ray_direction = pixel_sample - camera_center;

  let ray = Ray(camera_center, ray_direction);
  return ray;
}

fn pixel_sample_square(seed: ptr<function, u32>) -> vec3<f32> {
  let px = -0.5 + randFloat(seed);
  let py = -0.5 + randFloat(seed);
  return (px * pixel_delta_u) + (py * pixel_delta_v);
}

fn write_color(pixel_color: vec3<f32>, x: i32, y: i32) {
  let scale = 1.0 / f32(samples_per_pixel);
  let adjusted_color = pixel_color * scale;
  textureStore(texture, vec2<i32>(x, y), vec4<f32>(adjusted_color, 1.0));
}

@compute
@workgroup_size(${maxWorkgroupDimension}, ${maxWorkgroupDimension}, 1)
fn computeMain(@builtin(global_invocation_id) local_id: vec3<u32>) {
  var seed = local_id.x + local_id.y * ${imageWidth};
  
  let i = f32(local_id.x);
  let j = f32(local_id.y);
  
  var pixel_color = vec3<f32>(0.0, 0.0, 0.0);
  
  for (var sample = 0; sample < samples_per_pixel; sample += 1) {
      let r = get_ray(i, j, &seed);
      pixel_color += ray_color(r, &seed);
  }
  
  write_color(pixel_color, i32(local_id.x), i32(local_id.y));  
}
    `,
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

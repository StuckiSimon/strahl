import { buildBvhShader } from "./bvh";

type Params = {
  bvhParams: Parameters<typeof buildBvhShader>[0];
};

export function buildDenoisePassShader({ bvhParams }: Params) {
  return /* wgsl */ `
alias Color = vec3f;

override wgSize: u32 = 16;
override imageWidth: u32 = 512;
override imageHeight: u32 = 512;

struct Material {
  baseWeight: f32,
  baseColor: Color,
  // todo: switch order of baseDiffuseRoughness and baseMetalness
  baseDiffuseRoughness: f32,
  baseMetalness: f32,
  specularWeight: f32,
  specularColor: Color,
  specularRoughness: f32,
  specularAnisotropy: f32,
  specularRotation: f32,
  specularIor: f32,
  coatWeight: f32,
  coatColor: Color,
  coatRoughness: f32,
  coatRoughnessAnisotropy: f32,
  coatIor: f32,
  coatDarkening: f32,
  emissionLuminance: f32,
  emissionColor: Color,
  thinFilmThickness: f32,
  thinFilmIOR: f32,
}

struct UniformData {
  invProjectionMatrix: mat4x4<f32>,
  cameraWorldMatrix: mat4x4<f32>,
  invModelMatrix: mat4x4<f32>,
  seedOffset: u32,
  priorSamples: u32,
  samplesPerPixel: u32,
  sunDirection: vec3f,
  skyPower: f32,
  skyColor: Color,
  sunPower: f32,
  sunAngularSize: f32,
  sunColor: Color,
  clearColor: Color,
  // bool is not supported in uniform
  enableClearColor: i32,
  maxRayDepth: i32,
  objectDefinitionLength: i32,
  // 0 -> normal
  // 1 -> albedo
  mode: i32,
}

// Use due to 16 bytes alignment of vec3
struct IndicesPackage {
  x: i32,
  y: i32,
  z: i32,
}

@group(0) @binding(0) var<storage, read> positions: array<array<vec3f, 2>>;
// todo: Check when i16 is supported
@group(0) @binding(1) var<storage, read> indices: array<IndicesPackage>;

@group(0) @binding(2) var<storage, read> bounds: array<array<vec4f, 2>>;
@group(0) @binding(3) var<storage, read> contents: array<BinaryBvhNodeInfo>;

@group(0) @binding(5) var<storage, read> indirectIndices: array<u32>;

@group(0) @binding(6) var<storage, read> objectDefinitions: array<ObjectDefinition>;

@group(0) @binding(7) var<storage, read> materials: array<Material>;

@group(1) @binding(0) var texture: texture_storage_2d<rgba32float, write>;

@group(1) @binding(1) var readTexture: texture_storage_2d<rgba32float, read>;

@group(1) @binding(2) var<uniform> uniformData: UniformData;

@group(1) @binding(3) var<storage, read_write> hdrColor: array<vec4f>;

const MINIMUM_FLOAT_EPSILON = 1e-8;
const FLT_EPSILON = 1.1920929e-7;
const PI = 3.1415926535897932;
const PI_INVERSE = 1.0 / PI;

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
};

struct ObjectDefinition {
  start: u32,
  count: u32,
  material: MaterialDefinition,
}

struct MaterialDefinition {
  index: u32,
}

struct HitRecord {
  point: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  frontFace: bool,
  material: MaterialDefinition,
}

struct Triangle {
  Q: vec3<f32>,
  u: vec3<f32>,
  v: vec3<f32>,
  material: MaterialDefinition,
  normal0: vec3<f32>,
  normal1: vec3<f32>,
  normal2: vec3<f32>,
}

struct Interval {
  min: f32,
  max: f32,
}

struct BinaryBvhNodeInfo {
  // 0-16: isLeaf, 17-31: splitAxis|triangleCount
  x: u32,
  // rightIndex|triangleOffset
  y: u32,
}

fn nearZero(v: vec3f) -> bool {
  let epsilon = vec3f(MINIMUM_FLOAT_EPSILON);
  return any(abs(v) < epsilon);
}

fn sqr(x: f32) -> f32 {
  return x * x;
}

fn maxVec3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

fn rayAt(ray: Ray, t: f32) -> vec3<f32> {
  return ray.origin + t * ray.direction;
}

fn lengthSquared(v: vec3<f32>) -> f32 {
  return dot(v, v);
}

// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L283
const PCG_INC = 2891336453u;
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L278
const PCG_MULTIPLIER = 747796405u;

// https://www.pcg-random.org/download.html#id1
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L1533
fn randomU32(seed: u32) -> u32 {
  let state = seed * PCG_MULTIPLIER + PCG_INC;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

const range = 1.0 / f32(0xffffffffu);

// Generate a random float in the range [0, 1).
fn randomF32(seed: ptr<function, u32>) -> f32 {
  *seed = randomU32(*seed);
  return f32(*seed - 1u) * range;
}

${buildBvhShader(bvhParams)}

const TRIANGLE_MIN_DISTANCE_THRESHOLD = 0.0005;
const TRIANGLE_MAX_DISTANCE_THRESHOLD = 10e37f;

fn getRayOutput(cameraRay: Ray, seed: ptr<function, u32>) -> vec4f {
  var hitRecord: HitRecord;
  var ray = cameraRay;

  var dW = ray.direction;
  var pW = ray.origin;

  
  // todo: handle setting t nicely
  hitRecord.t = TRIANGLE_MAX_DISTANCE_THRESHOLD;
  let hit = hittableListHit(ray, Interval(TRIANGLE_MIN_DISTANCE_THRESHOLD, TRIANGLE_MAX_DISTANCE_THRESHOLD), &hitRecord);

  // todo: consider normal handling

  if (!hit) {
    // todo: reconsider
    return vec4f(1);
  }

  let material = materials[hitRecord.material.index];

  // Surface Normal
  var NsW = hitRecord.normal;
    
  if (uniformData.mode == 0) {
    return vec4f(-NsW.x, NsW.y, -NsW.z, 1.0);
  } else {
    return vec4f(material.baseColor * material.baseWeight, 1.0);
  }
}

fn sampleTriangleFilter(xi: f32) -> f32 {
  return select(1.0 - sqrt(2.0 - 2.0 * xi), sqrt(2.0 * xi) - 1.0, xi < 0.5);
}

fn ndcToCameraRay(coord: vec2f, cameraWorld: mat4x4<f32>, invProjectionMatrix: mat4x4<f32>, seed: ptr<function, u32>) -> Ray {
  let lookDirection = cameraWorld * vec4f(0.0, 0.0, -1.0, 0.0);
  let nearVector = invProjectionMatrix * vec4f(0.0, 0.0, -1.0, 1.0);
  let near = abs(nearVector.z / nearVector.w);

  var origin = cameraWorld * vec4f(0.0, 0.0, 0.0, 1.0);
  
  var direction = invProjectionMatrix * vec4f(coord.x, -coord.y, 0.5, 1.0);
  direction /= direction.w;
  direction = cameraWorld * direction - origin;

  origin += vec4f(direction.xyz * near / dot(direction, lookDirection), 0);

  return Ray(
    origin.xyz,
    direction.xyz
  );
}

fn getPixelJitter(seed: ptr<function, u32>) -> vec2f {
  let jitterX = 0.5 * sampleTriangleFilter(randomF32(seed));
  let jitterY = 0.5 * sampleTriangleFilter(randomF32(seed));
  return vec2f(jitterX, jitterY);
}

@compute
@workgroup_size(wgSize, wgSize, 1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  var seed = globalId.x + globalId.y * imageWidth;
  seed ^= uniformData.seedOffset;

  let pixelOrigin = vec2f(f32(globalId.x), f32(globalId.y));

  let pixel = pixelOrigin;
  let ndc = -1.0 + 2.0 * pixel / vec2f(f32(imageWidth), f32(imageHeight));
  
  var ray = ndcToCameraRay(ndc, uniformData.invModelMatrix * uniformData.cameraWorldMatrix, uniformData.invProjectionMatrix, &seed);
  ray.direction = normalize(ray.direction);

  let output = getRayOutput(ray, &seed);
  hdrColor[i32(globalId.x) + i32(globalId.y) * i32(imageWidth)] = output;
}
  `;
}

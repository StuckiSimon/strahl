alias Color = vec3<f32>;

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
}

@group(0) @binding(0) var<storage, read_write> positions: array<f32>;
// todo: Check when i16 is supported
@group(0) @binding(1) var<storage, read_write> indices: array<i32>;

@group(0) @binding(2) var<storage, read_write> bounds: array<f32>;
@group(0) @binding(3) var<storage, read_write> contents: array<BinaryBvhNodeInfo>;

@group(0) @binding(4) var<storage, read_write> normals: array<f32>;

@group(0) @binding(5) var<storage, read_write> indirectIndices: array<u32>;

@group(0) @binding(6) var<storage, read_write> objectDefinitions: array<ObjectDefinition>;

@group(0) @binding(7) var<storage, read_write> materials: array<Material>;

@group(1) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(1) var readTexture: texture_storage_2d<rgba8unorm, read>;

@group(1) @binding(2) var<uniform> uniformData: UniformData;

// todo: This should not be hardcoded
const indicesLength = 12636;

const maxDepth = 10;

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

fn identical(v1: vec3f, v2: vec3f) -> bool {
  return all(v1 == v2);
}

// Based on MaterialX ShaderGen implementation, which is in turn
// based on the OSL implementation of Oren-Nayar diffuse, which is in turn
// based on https://mimosa-pudica.net/improved-oren-nayar.html.
fn orenNayarDiffuse(L: vec3f, V: vec3f, N: vec3f, NdotL: f32, roughness: f32) -> f32 {
  let LdotV = clamp(dot(L, V), MINIMUM_FLOAT_EPSILON, 1.0);
  let NdotV = clamp(dot(N, V), MINIMUM_FLOAT_EPSILON, 1.0);
  let s = LdotV - NdotL * NdotV;
  let stinv = select(0.0, s / max(NdotL, NdotV), s > 0.0f);

  let sigma2 = pow(roughness * PI, 2);
  let A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
  let B = 0.45 * sigma2 / (sigma2 + 0.09);

  return A + B * stinv;
}

// Compute the average of an anisotropic alpha pair.
fn averageAlpha(alpha: vec2<f32>) -> f32 {
  return sqrt(alpha.x * alpha.y);
}

fn sqr(x: f32) -> f32 {
  return x * x;
}

fn maxVec3(v: vec3f) -> f32 {
  return max(v.x, max(v.y, v.z));
}

// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// Appendix B.2 Equation 13
fn ggxNDF(H: vec3f, alpha: vec2<f32>) -> f32 {
  let He = H.xy / alpha;
  let denom = dot(He, He) + (H.z*H.z);
  return 1.0 / (PI * alpha.x * alpha.y * denom * denom);
}

// todo: merge with ggxNDF
fn ggxNDFV2(H: vec3f, alpha: vec2<f32>) -> f32 {
  let safeAlpha = clamp(alpha, vec2(DENOM_TOLERANCE, DENOM_TOLERANCE), vec2(1.0, 1.0));
  let Ddenom = PI * safeAlpha.x * safeAlpha.y * sqr(sqr(H.x/safeAlpha.x) + sqr(H.y/safeAlpha.y) + sqr(H.z));
  return 1.0 / max(Ddenom, DENOM_TOLERANCE);
}

// GGX NDF sampling routine, as described in
// "Sampling Visible GGX Normals with Spherical Caps", Dupuy et al., HPG 2023.
// NB, this assumes wiL is in the +z hemisphere, and returns a sampled micronormal in that hemisphere.
fn ggxNDFSample(wiL: vec3f, alpha: vec2<f32>, seed: ptr<function, u32>) -> vec3f {
  let Xi = vec2f(randomF32(seed), randomF32(seed));
  var V = wiL;
  
  V = normalize(vec3f(V.xy * alpha, V.z));

  let phi = 2.0 * PI * Xi.x;
  let z = (1.0 - Xi.y) * (1.0 + V.z) - V.z;
  let sinTheta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
  let x = sinTheta * cos(phi);
  let y = sinTheta * sin(phi);
  let c = vec3f(x, y, z);

  var H = c + V;

  H = normalize(vec3f(H.xy * alpha, H.z));

  return H;
}

fn ggxNDFEval(m: vec3f, alpha: vec2f) -> f32 {
  let ax = max(alpha.x, DENOM_TOLERANCE);
  let ay = max(alpha.y, DENOM_TOLERANCE);
  let Ddenom = PI * ax * ay * sqr(sqr(m.x/ax) + sqr(m.y/ay) + sqr(m.z));
  return 1.0 / max(Ddenom, DENOM_TOLERANCE);
}

fn ggxLambda(w: vec3f, alpha: vec2f) -> f32 {
  if (abs(w.z) < FLT_EPSILON) {
    return 0.0;
  }
  return (-1.0 + sqrt(1.0 + (sqr(alpha.x*w.x) + sqr(alpha.y*w.y))/sqr(w.z))) / 2.0;
}

fn ggxG1(w: vec3f, alpha: vec2f) -> f32 {
  return 1.0 / (1.0 + ggxLambda(w, alpha));
}

fn ggxG2(woL: vec3f, wiL: vec3f, alpha: vec2f) -> f32 {
  return 1.0 / (1.0 + ggxLambda(woL, alpha) + ggxLambda(wiL, alpha));
}

// Height-correlated Smith masking-shadowing
// http://jcgt.org/published/0003/02/03/paper.pdf
// Equations 72 and 99
fn ggxSmithG2(NdotL: f32, NdotV: f32, alpha: f32) -> f32 {
  let alpha2 = alpha * alpha;
  let lambdaL = sqrt(alpha2 + (1.0 - alpha2) * (NdotL * NdotL));
  let lambdaV = sqrt(alpha2 + (1.0 - alpha2) * (NdotV * NdotV));
  return 2.0 / (lambdaL / NdotL + lambdaV / NdotV);
}

// todo: consider alternative DIRECTIONAL_ALBEDO_METHOD
fn ggxDirAlbedo(NdotV: f32, alpha: f32, F0: vec3f, F90: vec3f) -> vec3f {
  // Rational quadratic fit to Monte Carlo data for GGX directional albedo.
  let x = NdotV;
  let y = alpha;
  let x2 = x * x;
  let y2 = y * y;
  let r = vec4(0.1003, 0.9345, 1.0, 1.0) +
          vec4(-0.6303, -2.323, -1.765, 0.2281) * x +
          vec4(9.748, 2.229, 8.263, 15.94) * y +
          vec4(-2.038, -3.748, 11.53, -55.83) * x * y +
          vec4(29.34, 1.424, 28.96, 13.08) * x2 +
          vec4(-8.245, -0.7684, -7.507, 41.26) * y2 +
          vec4(-26.44, 1.436, -36.11, 54.9) * x2 * y +
          vec4(19.99, 0.2913, 15.86, 300.2) * x * y2 +
          vec4(-5.448, 0.6286, 33.37, -285.1) * x2 * y2;
  let AB = clamp(r.xy / r.zw, vec2(0.0, 0.0), vec2(1.0, 1.0));
  return F0 * AB.x + F90 * AB.y;
}

fn ggxDirAlbedoFloat(NdotV: f32, alpha: f32, F0: f32, F90: f32) -> f32 {
  return ggxDirAlbedo(NdotV, alpha, vec3f(F0), vec3f(F90)).x;
}

// https://blog.selfshadow.com/publications/turquin/ms_comp_final.pdf
// Equations 14 and 16
fn ggxEnergyCompensation(NdotV: f32, alpha: f32, Fss: vec3f) -> vec3f {
  let Ess = ggxDirAlbedoFloat(NdotV, alpha, 1.0, 1.0);
  return 1.0 + Fss * (Ess - 1.0) / Ess;
}

struct BsdfResponse {
  response: vec3f,
  throughput: vec3f,
  thickness: f32,
  ior: f32,
}

struct FresnelData {
  model: i32,
  ior: vec3f,
  extinction: vec3f,
  F0: vec3f,
  F90: vec3f,
  exponent: f32,
  thinFilmThickness: f32,
  thinFilmIOR: f32,
  refraction: bool,
}

const FRESNEL_MODEL_SCHLICK = 2;

fn initFresnelSchlick(F0: vec3f, F90: vec3f, exponent: f32) -> FresnelData {
  return FresnelData(
    FRESNEL_MODEL_SCHLICK,
    vec3f(0.0),
    vec3f(0.0),
    F0,
    F90,
    exponent,
    0.0,
    0.0,
    false
  );
}

fn fresnelSchlick(cosTheta: f32, F0: vec3f, F90: vec3f, exponent: f32) -> vec3f {
  let x = clamp(1.0 - cosTheta, 0.0, 1.0);
  return mix(F0, F90, pow(x, exponent));
}

fn fresnelSchlickV2(F0: vec3f, mu: f32) -> vec3f {
  return F0 + pow(1.0 - mu, 5.0) * (vec3f(1.0) - F0);
}

fn computeFresnel(cosTheta: f32, fd: FresnelData) -> vec3f {
  // todo: implement other models (dielectric, conductor, airy)
  if (fd.model == FRESNEL_MODEL_SCHLICK) {
    return fresnelSchlick(cosTheta, fd.F0, fd.F90, fd.exponent);
  }
  
  return vec3f(0.0);
}

fn fresnelF82Tint(mu: f32, F0: vec3f, f82Tint: vec3f) -> vec3f {
  let muBar = 1.0/7.0;
  let denom = muBar * pow(1.0 - muBar, 6);
  let fSchlickBar = fresnelSchlickV2(F0, muBar);
  let fSchlick = fresnelSchlickV2(F0, mu);
  return fSchlick - mu * pow(1.0 - mu, 6.0) * (vec3f(1.0) - f82Tint) * fSchlickBar / denom;
}


fn rayAt(ray: Ray, t: f32) -> vec3<f32> {
  return ray.origin + t * ray.direction;
}

fn lengthSquared(v: vec3<f32>) -> f32 {
  return dot(v, v);
}

// CODE#RNG
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L283
const PCG_INC = 2891336453u;
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L278
const PCG_MULTIPLIER = 747796405u;

// https://www.pcg-random.org/download.html#id1
// See https://github.com/imneme/pcg-c/blob/83252d9c23df9c82ecb42210afed61a7b42402d7/include/pcg_variants.h#L1533
fn randomU32(seed: u32) -> u32 {
  let state = seed * PCG_MULTIPLIER + PCG_INC;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return u32((word >> 22u) ^ word);
}

const range = 1.0 / f32(0xffffffffu);

// Generate a random float in the range [0, 1).
fn randomF32(seed: ptr<function, u32>) -> f32 {
  *seed = randomU32(*seed);
  return f32(*seed - 1u) * range;
}

fn triangleHit(triangle: Triangle, ray: Ray, rayT: Interval, hitRecord: ptr<function, HitRecord>) -> bool {
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
  if (t < (rayT).min || t > (rayT).max) {
    return false;
  }

  (*hitRecord).t = t;
  (*hitRecord).point = rayAt(ray, t);
  (*hitRecord).normal = normalize(triangle.normal0 * (1.0 - u - v) + triangle.normal1 * u + triangle.normal2 * v);

  (*hitRecord).material = triangle.material;

  return true;
}

// Based on https://github.com/gkjohnson/three-mesh-bvh/blob/master/src/gpu/glsl/bvh_ray_functions.glsl.js
fn intersectsBounds(ray: Ray, boundsMin: vec3<f32>, boundsMax: vec3<f32>, dist: ptr<function, f32>) -> bool {
  let invDir = vec3<f32>(1.0) / ray.direction;
  
  let tMinPlane = invDir * (boundsMin - ray.origin);
  let tMaxPlane = invDir * (boundsMax - ray.origin);

  let tMinHit = min(tMaxPlane, tMinPlane);
  let tMaxHit = max(tMaxPlane, tMinPlane);

  var t = max(tMinHit.xx, tMinHit.yz);
  let t0 = max(t.x, t.y);

  t = min(tMaxHit.xx, tMaxHit.yz);
  let t1 = min(t.x, t.y);

  (*dist) = max(t0, 0.0);

  return t1 >= (*dist);
}

fn intersectsBVHNodeBounds(ray: Ray, currNodeIndex: u32, dist: ptr<function, f32>) -> bool {
  // 2 because min+max, 4 because x,y,z + unused alpha
  let cni2 = currNodeIndex * 2u * 4;
  let boundsMin = vec3<f32>(bounds[cni2], bounds[cni2 + 1], bounds[cni2 + 2]);
  // Start at 4 because of unused alpha
  let boundsMax = vec3<f32>(bounds[cni2 + 4], bounds[cni2 + 5], bounds[cni2 + 6]);
  return intersectsBounds(ray, boundsMin, boundsMax, dist);
}

fn intersectTriangles(offset: u32, count: u32, ray: Ray, hitRecord: ptr<function, HitRecord>) -> bool {
  var found = false;
  var localDist = hitRecord.t;
  let l = offset + count;
  
  for (var i = offset; i < l; i += 1) {
    let vIndexOffset = indirectIndices[i] * 3u;

    let v1Index = indices[vIndexOffset];
    let v2Index = indices[vIndexOffset+1];
    let v3Index = indices[vIndexOffset+2];
    
    let x = vec3f(positions[v1Index*3], positions[v1Index*3+1], positions[v1Index*3+2]);
    let y = vec3f(positions[v2Index*3], positions[v2Index*3+1], positions[v2Index*3+2]);
    let z = vec3f(positions[v3Index*3], positions[v3Index*3+1], positions[v3Index*3+2]);
    
    let Q = x;
    let u = y - x;
    let v = z - x;
    
    var matchingObjectDefinition: ObjectDefinition = objectDefinitions[0];
    // todo: consider more elegant limit than 100 object parts
    for (var j = 0; j < 100; j++) {
      let objectDefinition = objectDefinitions[j];
      if (objectDefinition.start <= vIndexOffset && objectDefinition.start + objectDefinition.count >= vIndexOffset) {
        matchingObjectDefinition = objectDefinition;
        break;
      }
    }
    let materialDefinition = matchingObjectDefinition.material;

    let normalX = vec3f(normals[v1Index*3], normals[v1Index*3+1], normals[v1Index*3+2]);
    let normalY = vec3f(normals[v2Index*3], normals[v2Index*3+1], normals[v2Index*3+2]);
    let normalZ = vec3f(normals[v3Index*3], normals[v3Index*3+1], normals[v3Index*3+2]);
    
    let triangle = Triangle(Q, u, v, materialDefinition, normalX,normalY,normalZ);

    var tmpRecord: HitRecord;
    // todo: reuse rayT.min
    if (triangleHit(triangle, ray, Interval(0.0001, localDist), &tmpRecord)) {
      if (localDist < tmpRecord.t) {
        continue;
      }
      (*hitRecord) = tmpRecord;

      localDist = (*hitRecord).t;
      found = true;
    }
  }
  return found;
}

// todo: lower is better, maybe this should be set based on the tree depth
const maxBvhStackDepth = 60;

fn hittableListHit(ray: Ray, rayT: Interval, hitRecord: ptr<function, HitRecord>) -> bool {
  var tempRecord: HitRecord;
  var hitAnything = false;
  var closestSoFar = rayT.max;

  for (var i = 0; i < TRIANGLE_COUNT; i += 1) {
    let triangle = triangles[i];
    if (triangleHit(triangle, ray, Interval(rayT.min, closestSoFar), &tempRecord)) {
      hitAnything = true;
      closestSoFar = tempRecord.t;
      (*hitRecord) = tempRecord;
      (*hitRecord).material = triangle.material;
    }
  }


  // Inspired by https://github.com/gkjohnson/three-mesh-bvh/blob/master/src/gpu/glsl/bvh_ray_functions.glsl.js
  
  // BVH Intersection Detection
  var sPtr = 0;
  var stack: array<u32, maxBvhStackDepth> = array<u32, maxBvhStackDepth>();
  stack[sPtr] = 0u;

  while (sPtr > -1 && sPtr < maxBvhStackDepth) {
    let currNodeIndex = stack[sPtr];
    sPtr -= 1;

    var boundsHitDistance: f32;
    
    if (!intersectsBVHNodeBounds(ray, currNodeIndex, &boundsHitDistance) || boundsHitDistance > closestSoFar) {
      continue;
    }

    let boundsInfo = contents[currNodeIndex];
    let boundsInfoX = boundsInfo.x;
    let boundsInfoY = boundsInfo.y;

    let isLeaf = (boundsInfoX & 0xffff0000u) == 0xffff0000u;

    if (isLeaf) {
      let count = boundsInfoX & 0x0000ffffu;
      let offset = boundsInfoY;

      let found2 = intersectTriangles(
        offset,
        count,
        ray,
        hitRecord
      );
      if (found2) {
        closestSoFar = (*hitRecord).t;
      }
      
      hitAnything = hitAnything || found2;
    } else {
      // Left node is always the next node
      let leftIndex = currNodeIndex + 1u;
      let splitAxis = boundsInfoX & 0x0000ffffu;
      let rightIndex = boundsInfoY;

      let leftToRight = ray.direction[splitAxis] > 0.0;
      let c1 = select(rightIndex, leftIndex, leftToRight);
      let c2 = select(leftIndex, rightIndex, leftToRight);

      sPtr += 1;
      stack[sPtr] = c2;
      sPtr += 1;
      stack[sPtr] = c1;
    }
  }

  // todo: remove this demo code
  /*
  for (var i = 0; i < indicesLength; i += 3) {
    let v1Index = indices[i];
    let v2Index = indices[i+1];
    let v3Index = indices[i+2];
    
    let x = vec3f(positions[v1Index*3], positions[v1Index*3+1], positions[v1Index*3+2]);
    let y = vec3f(positions[v2Index*3], positions[v2Index*3+1], positions[v2Index*3+2]);
    let z = vec3f(positions[v3Index*3], positions[v3Index*3+1], positions[v3Index*3+2]);
    
    let Q = x;
    let u = y - x;
    let v = z - x;
    
    let triangle = Triangle(Q, u, v, defaultMaterial, defaultNormal,defaultNormal,defaultNormal);

    if (triangleHit(triangle, ray, Interval(rayT.min, closestSoFar), &tempRecord)) {
      hitAnything = true;
      closestSoFar = tempRecord.t;
      (*hitRecord) = tempRecord;
      (*hitRecord).material = triangle.material;
    }
  } */

  return hitAnything;
}

struct BouncingInfo {
  attenuation: Color,
  emission: Color,
}

struct Basis {
  nW: vec3f,
  tW: vec3f,
  bW: vec3f,
  baryCoords: vec3f,
}

const DENOM_TOLERANCE = 1.0e-10;
const RADIANCE_EPSILON = 1.0e-12;

fn safeNormalize(v: vec3f) -> vec3f {
  let len = length(v);
  return v/max(len, DENOM_TOLERANCE);
}

fn normalToTangent(N: vec3f) -> vec3f {
  var T: vec3f;
  if (abs(N.z) < abs(N.x)) {
    T = vec3f(N.z, 0.0, -N.x);
  } else {
    T = vec3f(0.0, N.z, -N.y);
  }
  return safeNormalize(T);
}

fn makeBasis(nWI: vec3f) -> Basis {
  let nW = safeNormalize(nWI);
  let tW = normalToTangent(nWI);
  let bW = cross(nWI, tW);
  return Basis(nW, tW, bW, vec3f(0.0));
}

fn makeBasisFull(nW: vec3f, tW: vec3f, baryCoords: vec3f) -> Basis {
  let nWo = safeNormalize(nW);
  let tWo = safeNormalize(tW);
  let bWo = cross(nWo, tWo);
  return Basis(nWo, tWo, bWo, baryCoords);
}

fn worldToLocal(vWorld: vec3f, basis: Basis) -> vec3f {
  return vec3f(dot(vWorld, basis.tW), dot(vWorld, basis.bW), dot(vWorld, basis.nW));
}

fn localToWorld(vLocal: vec3f, basis: Basis) -> vec3f {
  return basis.tW * vLocal.x + basis.bW * vLocal.y + basis.nW * vLocal.z;
}

struct LocalFrameRotation {
  M: mat2x2<f32>,
  Minv: mat2x2<f32>,
}

fn getLocalFrameRotation(angle: f32) -> LocalFrameRotation {
  if (angle == 0.0 || angle==2*PI) {
    let identity = mat2x2<f32>(1.0, 0.0, 0.0, 1.0);
    return LocalFrameRotation(identity, identity);
  } else {
    let cosRot = cos(angle);
    let sinRot = sin(angle);
    let M = mat2x2<f32>(cosRot, sinRot, -sinRot, cosRot);
    let Minv = mat2x2<f32>(cosRot, -sinRot, sinRot, cosRot);
    return LocalFrameRotation(M, Minv);
  }
}

fn localToRotated(vLocal: vec3f, rotation: LocalFrameRotation) -> vec3f {
  let xyRot = rotation.M * vLocal.xy;
  return vec3f(xyRot.x, xyRot.y, vLocal.z);
}

fn rotatedToLocal(vRotated: vec3f, rotation: LocalFrameRotation) -> vec3f {
  let xyLocal = rotation.Minv * vRotated.xy;
  return vec3f(xyLocal.x, xyLocal.y, vRotated.z);
}

struct LobeWeights {
  m: array<vec3f, NUM_LOBES>,
}

struct LobeAlbedos {
  m: array<Color, NUM_LOBES>,
}

struct LobeProbs {
  m: array<f32, NUM_LOBES>,
}

struct LobePDFs {
  m: array<f32, NUM_LOBES>,
}

struct LobeData {
  weights: LobeWeights,
  albedos: LobeAlbedos,
  probs: LobeProbs,
}

// todo: implement
fn placeholderBrdfAlbedo() -> Color {
  return Color(0.0, 0.0, 0.0);
}

fn specularNDFRoughness(material: Material) -> vec2f {
  let rsqr = material.specularRoughness * material.specularRoughness;
  let specularAnisotropyInv = 1.0 - material.specularAnisotropy;
  let alphaX = rsqr * sqrt(2.0/(1.0+(specularAnisotropyInv*specularAnisotropyInv)));
  let alphaY = (1.0 - material.specularAnisotropy) * alphaX;

  let minAlpha = 1.0e-4;
  return vec2f(max(alphaX, minAlpha), max(alphaY, minAlpha));
}

fn metalBrdfEvaluate(pW: vec3f, basis: Basis, winputL: vec3f, woutputL: vec3f, material: Material, pdfWoutputL: ptr<function, f32>) -> vec3f {
  if (winputL.z < DENOM_TOLERANCE || woutputL.z < DENOM_TOLERANCE) {
    (*pdfWoutputL) = PDF_EPSILON;
    return vec3f(0.0);
  }

  let rotation = getLocalFrameRotation(2*PI*material.specularRotation);
  let winputR = localToRotated(winputL, rotation);
  let woutputR = localToRotated(woutputL, rotation);

  let alpha = specularNDFRoughness(material);

  let mR = normalize(winputR + woutputR);

  let D = ggxNDFEval(mR, alpha);
  let DV = D * ggxG1(winputR, alpha) * max(0.0, dot(winputR, mR)) / max(DENOM_TOLERANCE, winputR.z);

  let dwhDwo = 1.0 / max(abs(4.0*dot(winputR, mR)), DENOM_TOLERANCE);
  (*pdfWoutputL) = max(PDF_EPSILON, DV * dwhDwo);

  let FnoFilm = fresnelF82Tint(abs(dot(winputR, mR)), material.baseWeight * material.baseColor, material.specularWeight * material.specularColor);

  // todo: thin film workflow

  let F = FnoFilm;

  let G2 = ggxG2(winputR, woutputR, alpha);

  return F * D * G2 * max(4.0*abs(woutputL.z)*abs(winputL.z), DENOM_TOLERANCE);
}

fn metalBrdfSample(pW: vec3f, basis: Basis, winputL: vec3f, material: Material, seed: ptr<function, u32>, woutputL: ptr<function, vec3f>, pdfWoutputL: ptr<function, f32>) -> vec3f {
  if (winputL.z < DENOM_TOLERANCE) {
    (*pdfWoutputL) = PDF_EPSILON;
    return vec3f(0.0);
  }

  let alpha = specularNDFRoughness(material);

  var rotation = getLocalFrameRotation(2*PI*material.specularRotation);
  var winputR = localToRotated(winputL, rotation);

  let mR = ggxNDFSample(winputR, alpha, seed);

  let woutputR = -winputR + 2.0*dot(winputR, mR)*mR;
  if (winputR.z * woutputR.z < FLT_EPSILON) {
    return vec3f(0.0);
  }
  (*woutputL) = rotatedToLocal(woutputR, rotation);

  let D = ggxNDFV2(mR, alpha);
  let DV = D * ggxG1(winputR, alpha) * max(0.0, dot(winputR, mR)) / max(DENOM_TOLERANCE, winputR.z); // todo: should latter max term use abs for .z?
  
  let dwhDwo = 1.0 / max(abs(4.0*dot(winputR, mR)), DENOM_TOLERANCE);
  (*pdfWoutputL) = max(PDF_EPSILON, DV * dwhDwo);

  // todo: implement thin film workflow
  let F_nofilm = fresnelF82Tint(abs(dot(winputR, mR)), material.baseWeight * material.baseColor, material.specularWeight * material.specularColor);
  let F = F_nofilm;
  
  let G2 = ggxG2(winputR, woutputR, alpha);
  
  return F * D * G2 / max(4.0*abs(woutputL.z)*abs(winputL.z), DENOM_TOLERANCE);
}

fn metalBrdfAlbedo(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, seed: ptr<function, u32>) -> Color {
  if (winputL.z < DENOM_TOLERANCE) {
    return vec3f(0.0);
  }

  let numSamples = 1;
  var albedo = vec3f(0.0);
  for (var n=0; n<numSamples; n+=1) {
    var woutputL: vec3f;
    var pdfWoutputL: f32;
    var f = metalBrdfSample(pW, basis, winputL, material, seed, &woutputL, &pdfWoutputL);
    if (length(f) > RADIANCE_EPSILON) {
      albedo += f * abs(woutputL.z) / max(PDF_EPSILON, pdfWoutputL);
    }
  }

  albedo /= f32(numSamples);
  return albedo;
}

fn diffuseBrdfAlbedo(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, seed: ptr<function, u32>) -> vec3f {
    if (winputL.z < DENOM_TOLERANCE) {
    return vec3f(0.0);
  }
  return material.baseWeight * material.baseColor;
}

// https://github.com/AcademySoftwareFoundation/MaterialX/blob/main/libraries/pbrlib/genglsl/lib/mx_microfacet_diffuse.glsl
fn fujiiMaterialX(albedo: vec3f, roughness: f32, V: vec3f, L: vec3f) -> vec3f {
  let NdotV = V.z;
  let NdotL = L.z;
  let s = dot(L, V) - NdotV * NdotL;
  let stinv = select(0.0, s / max(NdotL, NdotV), s > 0.0f);
  let sigma = roughness;
  let sigma2 = sqr(sigma);
  let A = 1.0 - 0.5 * (sigma2 / (sigma2 + 0.33));
  let B = 0.45 * sigma2 / (sigma2 + 0.09);
  return albedo * NdotL / PI * (A + B * stinv);
}

fn diffuseBrdfEvalImplementation(woutputL: vec3f, winputL: vec3f, material: Material) -> vec3f {
  let albedo = material.baseWeight * material.baseColor;
  let V = winputL;
  let L = woutputL;
  let NdotL = max(FLT_EPSILON, abs(L.z));
  
  return fujiiMaterialX(albedo, material.baseDiffuseRoughness, V, L) / NdotL;
}

fn diffuseBrdfEvaluate(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, woutputL: vec3f, pdfWoutputL: ptr<function, f32>) -> vec3f {
  if (winputL.z < DENOM_TOLERANCE || woutputL.z < DENOM_TOLERANCE) {
    return vec3f(0.0);
  }
  (*pdfWoutputL) = pdfHemisphereCosineWeighted(woutputL);
  return diffuseBrdfEvalImplementation(winputL, woutputL, material);
}

fn diffuseBrdfSample(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, woutputL: ptr<function, vec3f>, pdfWoutputL: ptr<function, f32>, seed: ptr<function, u32>) -> vec3f {
  if (winputL.z < DENOM_TOLERANCE) {
    return vec3f(0.0);
  }
  (*woutputL) = sampleHemisphereCosineWeighted(pdfWoutputL, seed);
  return diffuseBrdfEvalImplementation(winputL, *woutputL, material);
}


struct WeightsAndAlbedo {
  weights: LobeWeights,
  albedos: LobeAlbedos,
}

fn openPbrLobeWeights(pW: vec3f, basis: Basis, winputL: vec3f, material: Material, seed: ptr<function, u32>) -> WeightsAndAlbedo {
  let F = 0.0; // todo: move to material definition fuzzWeight
  let C = material.coatWeight;
  let M = material.baseMetalness;
  let T = 0.0; // todo: move to material definition transmissionWeight
  let S = 0.0; // todo: move to material definition subsurfaceWeight

  let coated = C > 0.0;
  let metallic = M > 0.0;
  let fullyMetallic = M == 1.0;
  let transmissive = T > 0.0;
  let fullyTransmissive = T == 1.0;
  let subsurfaced = S > 0.0;
  let fullySubsurfaced = S == 1.0;

  var albedos = LobeAlbedos();
  albedos.m[ID_COAT_BRDF] = select(vec3f(0.0), placeholderBrdfAlbedo(), coated);
  albedos.m[ID_META_BRDF] = select(vec3f(0.0), metalBrdfAlbedo(material, pW, basis, winputL, seed), metallic);
  albedos.m[ID_SPEC_BRDF] = select(vec3f(0.0), placeholderBrdfAlbedo(), !fullyMetallic);
  albedos.m[ID_SPEC_BTDF] = select(vec3f(0.0), placeholderBrdfAlbedo(), !fullyMetallic && transmissive);
  albedos.m[ID_DIFF_BRDF] = select(vec3f(0.0), diffuseBrdfAlbedo(material, pW, basis, winputL, seed), !fullyMetallic && !fullyTransmissive && !fullySubsurfaced);
  albedos.m[ID_SSSC_BTDF] = select(vec3f(0.0), placeholderBrdfAlbedo(), !fullyMetallic && !fullyTransmissive && subsurfaced);

  var weights = LobeWeights();

  weights.m[ID_FUZZ_BRDF] = vec3f(0.0); // todo: check

  let wCoatedBase = vec3f(1.0); // todo: check 

  weights.m[ID_COAT_BRDF] = wCoatedBase * C;

  // todo: implement coat workflow
  let baseDarkening = vec3f(1.0); // todo: check
  let materialCoatColor = vec3f(1.0); // todo: move to material definition (coat_color)
  let wBaseSubstrate = wCoatedBase * mix(vec3f(1.0), baseDarkening * materialCoatColor * (vec3(1.0) - albedos.m[ID_COAT_BRDF]), C);

  weights.m[ID_META_BRDF] = wBaseSubstrate * M;

  let wDielectricBase = wBaseSubstrate * vec3f(max(0.0, 1.0 - M));

  weights.m[ID_SPEC_BRDF] = wDielectricBase;

  weights.m[ID_SPEC_BTDF] = wDielectricBase * T;

  let wOpaqueDielectricBase = wDielectricBase * (1.0 - T);

  weights.m[ID_SSSC_BTDF] = wOpaqueDielectricBase * S;

  weights.m[ID_DIFF_BRDF] = wOpaqueDielectricBase * (1.0 - S) * (vec3f(1.0) - albedos.m[ID_SPEC_BRDF]);

  return WeightsAndAlbedo(
    weights,
    albedos
  );
}


fn openPbrLobeProbabilities(weights: LobeWeights, albedos: LobeAlbedos) -> LobeProbs {
  var probs = LobeProbs();
  var Wtotal = 0.0;
  for (var lobeId = 0; lobeId <= NUM_LOBES; lobeId += 1) {
    probs.m[lobeId] = length(weights.m[lobeId] * albedos.m[lobeId]);
    Wtotal += probs.m[lobeId];
  }
  Wtotal = max(DENOM_TOLERANCE, Wtotal);
  for (var lobeId = 0; lobeId <= NUM_LOBES; lobeId += 1) {
    probs.m[lobeId] /= Wtotal;
  }
  return probs;
}

fn openPbrPrepare(pW: vec3f, basis: Basis, winputL: vec3f, material: Material, seed: ptr<function, u32>) -> LobeData {
  let weightsAndAlbedo = openPbrLobeWeights(pW, basis, winputL, material, seed);
  let probs = openPbrLobeProbabilities(weightsAndAlbedo.weights, weightsAndAlbedo.albedos);

  return LobeData(
    weightsAndAlbedo.weights,
    weightsAndAlbedo.albedos,
    probs,
  );
}

const PDF_EPSILON = 1.0e-6;
const RAY_OFFSET = 1.0e-4;

fn pdfHemisphereCosineWeighted(wiL: vec3f) -> f32 {
  if (wiL.z <= PDF_EPSILON) {
    return PDF_EPSILON / PI;
  }
  return wiL.z / PI;
}

fn sampleHemisphereCosineWeighted(pdf: ptr<function, f32>, seed: ptr<function, u32>) -> vec3f {
  let r = sqrt(randomF32(seed));
  let theta = 2.0 * PI * randomF32(seed);
  let x = r * cos(theta);
  let y = r * sin(theta);
  let z = sqrt(max(0.0, 1.0 - x*x - y*y));
  (*pdf) = max(PDF_EPSILON, abs(z) / PI);
  return vec3f(x, y, z);
}

fn skyPdf(woutputL: vec3f, woutputWs: vec3f) -> f32 {
  return pdfHemisphereCosineWeighted(woutputL);
}

fn sunPdf(woutputL: vec3f, woutputW: vec3f) -> f32 {
  let thetaMax = uniformData.sunAngularSize * PI/180.0;
  if (dot(woutputW, uniformData.sunDirection) < cos(thetaMax))  {
    return 0.0;
  }
  let solidAngle = 2.0 * PI * (1.0 - cos(thetaMax));
  return 1.0 / solidAngle;
}

fn sunTotalPower() -> f32 {
  let thetaMax = uniformData.sunAngularSize * PI/180.0;
  let solidAngle = 2.0 * PI * (1.0 - cos(thetaMax));
  return length(uniformData.sunPower * uniformData.sunColor) * solidAngle;
}

fn skyTotalPower() -> f32 {
  return length(uniformData.skyPower * uniformData.skyColor) * 2.0 * PI;
}

fn sunRadiance(woutputW: vec3f) -> vec3f {
  let thetaMax = uniformData.sunAngularSize * PI/180.0;
  if (dot(woutputW, uniformData.sunDirection) < cos(thetaMax)) {
    return vec3f(0.0);
  }
  return uniformData.sunPower * uniformData.sunColor;
}

fn skyRadiance() -> vec3f {
  return uniformData.skyPower * uniformData.skyColor;
}

fn lightPdf(shadowW: vec3f, basis: Basis) -> f32 {
  let shadowL = worldToLocal(shadowW, basis);
  let pdfSky = skyPdf(shadowL, shadowW);
  let pdfSun = sunPdf(shadowL, shadowW);
  let wSun = sunTotalPower();
  let wSky = skyTotalPower();
  let pSun = wSun / (wSun + wSky);
  let pSky = max(0.0, 1.0 - pSun);
  let lightPdf = pSun * pdfSun + pSky * pdfSky;
  
  return lightPdf;
}

fn powerHeuristic(a: f32, b: f32) -> f32 {
  return pow(a, 2) / max(DENOM_TOLERANCE, pow(a, 2) + pow(b, 2));
}

fn brdfSamplePlaceholder() -> vec3f {
  return vec3f(0.0);
}

fn brdfEvaluatePlaceholder() -> vec3f {
  return vec3f(0.0);
}

fn openpbrBsdfEvaluateLobes(pW: vec3f, basis: Basis, material:Material, winputL: vec3f, woutputL: ptr<function, vec3f>, skipLobeId: i32, lobeData: LobeData, pdfs: ptr<function, LobePDFs>, seed: ptr<function, u32>
) -> vec3f {
  var f = vec3f(0.0);
  if (skipLobeId != ID_FUZZ_BRDF && lobeData.probs.m[ID_FUZZ_BRDF] > 0.0) {
    f += vec3f(0.0);
  } else if (skipLobeId != ID_COAT_BRDF && lobeData.probs.m[ID_COAT_BRDF] > 0.0) {
    f += lobeData.weights.m[ID_COAT_BRDF] * brdfEvaluatePlaceholder();
  } else if (skipLobeId != ID_META_BRDF && lobeData.probs.m[ID_META_BRDF] > 0.0) {
    //f += metalBrdfSample(pW, basis, winputL, material, seed, woutputL, &pdfs.m[ID_META_BRDF]); // fixme: evaluate???
    f += metalBrdfEvaluate(pW, basis, winputL, *woutputL, material, &pdfs.m[ID_META_BRDF]);
  } else if (skipLobeId != ID_SPEC_BRDF && lobeData.probs.m[ID_SPEC_BRDF] > 0.0) {
    f += lobeData.weights.m[ID_SPEC_BRDF] * brdfEvaluatePlaceholder();
  } else if (skipLobeId != ID_DIFF_BRDF && lobeData.probs.m[ID_DIFF_BRDF] > 0.0) {
    f += lobeData.weights.m[ID_DIFF_BRDF] * diffuseBrdfEvaluate(material, pW, basis, winputL, *woutputL, &pdfs.m[ID_DIFF_BRDF]);
  }

  let evalSpecBtdf = skipLobeId != ID_SPEC_BTDF && lobeData.probs.m[ID_SPEC_BTDF] > 0.0;
  let evalSsscBtdf = skipLobeId != ID_SSSC_BTDF && lobeData.probs.m[ID_SSSC_BTDF] > 0.0;
  let evalTransmission = evalSpecBtdf || evalSsscBtdf;
  if (evalTransmission) {
    // todo: implement
  }

  return f;
}

fn openpbrBsdfTotalPdf(pdfs: LobePDFs, lobeData: LobeData) -> f32 {
  var pdfWoutputL = 0.0;
  for (var lobeId = 0; lobeId < NUM_LOBES; lobeId += 1) {
    pdfWoutputL += lobeData.probs.m[lobeId] * pdfs.m[lobeId];
  }
  return pdfWoutputL;
}

const ID_FUZZ_BRDF = 0;
const ID_COAT_BRDF = 1;
const ID_META_BRDF = 2;
const ID_SPEC_BRDF = 3;
const ID_SPEC_BTDF = 4;
const ID_DIFF_BRDF = 5;
const ID_SSSC_BTDF = 6;
const NUM_LOBES    = 7;

fn sampleBsdf(pW: vec3f, basis: Basis, winputL: vec3f, lobeData: LobeData, material: Material, woutputL: ptr<function, vec3f>, pdfWoutputL: ptr<function, f32>, seed: ptr<function, u32>) -> vec3f {
  let X = randomF32(seed);
  var CDF = 0.0;

  for (var lobeId = 0; lobeId < NUM_LOBES; lobeId += 1) {
    CDF += lobeData.probs.m[lobeId];
    if (X < CDF) {
      var pdfLobe: f32;
      var fLobe: vec3f;
      if (lobeId == ID_FUZZ_BRDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_COAT_BRDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_META_BRDF) {
        fLobe = metalBrdfSample(pW, basis, winputL, material, seed, woutputL, &pdfLobe);
      }
      else if (lobeId == ID_SPEC_BRDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_SPEC_BTDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_SSSC_BTDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_DIFF_BRDF) {
        fLobe = diffuseBrdfSample(material, pW, basis, winputL, woutputL, &pdfLobe, seed);
        }
      else { break; }

      var pdfs: LobePDFs;
      var skipLobeId = lobeId;
      var f = openpbrBsdfEvaluateLobes(pW, basis, material, winputL, woutputL, skipLobeId, lobeData, &pdfs, seed);
      f += lobeData.weights.m[lobeId] * fLobe;

      pdfs.m[lobeId] = pdfLobe;
      (*pdfWoutputL) = openpbrBsdfTotalPdf(pdfs, lobeData);
      
      let transmitted = woutputL.z * winputL.z < 0.0;
      let transmittedInside = transmitted && woutputL.z < 0.0;
      if (!transmittedInside) {
        return f;
      }

      // todo: volume

      return f;
    }
  }

  (*pdfWoutputL) = 1.0;
  return vec3f(0);
}

fn rayColor(cameraRay: Ray, seed: ptr<function, u32>) -> vec3<f32> {
  var hitRecord: HitRecord;
  var ray = cameraRay;

  var throughput = vec3f(1.0);
  var L = vec3f(0.0);
  var bsdfPdfContinuation = 1.0;

  var dW = ray.direction;
  var pW = ray.origin;

  var basis: Basis;

  var inDielectric = false;

  for (var i = 0; i < maxDepth; i += 1) {
    hitRecord.t = 99999999999999999999.0;
    let hit = hittableListHit(ray, Interval(0.001, 0xfffffffffffffff), &hitRecord);

    // todo: consider normal handling

    if (!hit) {
      // did not hit anything until infinity

      var misWeightLight = 1.0;
      
      if (i > 0) {
        let lightPdf = lightPdf(
          dW,
          basis
        );
        misWeightLight = powerHeuristic(bsdfPdfContinuation, lightPdf);
      } else {
        if (uniformData.enableClearColor == 1) {
          return uniformData.clearColor;
      }
      }
      L += throughput * misWeightLight * (sunRadiance(dW) + skyRadiance());
      break;
    }

    let material = materials[hitRecord.material.index];

    // Surface Normal
    var NsW = hitRecord.normal;
    // Geometric Normal todo: distinguish between geometric and shading normal
    var NgW = NsW;
    // Tangent
    let TsW = normalToTangent(NsW);
    let baryCoords = vec3f(0.0); // todo: implement
    
    pW = hitRecord.point;

    if (
      (inDielectric && dot(NsW, dW) < 0.0) ||
      (!inDielectric && dot(NsW, dW) > 0.0)) {
      NsW = -NsW;
    }

    if (dot(NgW, NsW) < 0.0) {
      NgW = -NgW;
    }

    basis = makeBasisFull(NgW, TsW, baryCoords);

    let winputW = -dW;
    let winputL = worldToLocal(winputW, basis);

    let lobeData = openPbrPrepare(pW, basis, winputL, material, seed);

    var woutputL: vec3f;
    let f = sampleBsdf(pW, basis, winputL, lobeData, material, &woutputL, &bsdfPdfContinuation, seed);
    let woutputW = localToWorld(woutputL, basis);
    let surfaceThroughput = f / max(PDF_EPSILON, bsdfPdfContinuation) * abs(dot(woutputW, basis.nW));
    dW = woutputW;

    // todo: consider emission

    pW += NgW * sign(dot(dW, NgW)) * RAY_OFFSET;

    ray = Ray(pW, dW);

    var transmitted = dot(winputW, NgW) * dot(dW, NgW) < 0.0;
    if (transmitted) {
      inDielectric = !inDielectric;
    }

    if (!inDielectric && !transmitted) {
      // todo: 
    }

    throughput *= surfaceThroughput;

    // Russian Roulette
    if (maxVec3(throughput) < 1.0 && i > 1) {
      let q = max(0.0, 1.0 - maxVec3(throughput));
      if (randomF32(seed) < q) {
        break;
      }
      throughput /= 1.0 - q;
    }
  }

  return L;
}

fn writeColor(pixelColor: vec3<f32>, x: i32, y: i32, samples: i32) {
  let previousColor = textureLoad(readTexture, vec2<i32>(x, y)).xyz;
  let previousColorAdjusted = previousColor * f32(uniformData.priorSamples);
  let samplesPerPixel = uniformData.samplesPerPixel;
  let scale = 1.0 / f32(uniformData.priorSamples + samplesPerPixel);
  let adjustedColor = (pixelColor + previousColorAdjusted) * scale;
  textureStore(texture, vec2<i32>(x, y), vec4<f32>(adjustedColor, 1.0));
}

@must_use
fn identityMatrix() -> mat4x4<f32> {
  return mat4x4<f32>(
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
  );
}

fn sampleTriangleFilter(xi: f32) -> f32 {
  return select(1.0 - sqrt(2.0 - 2.0 * xi), sqrt(2.0 * xi) - 1.0, xi < 0.5);
}

// CODE#VIEWPROJECTION
fn ndcToCameraRay(coord: vec2f, cameraWorld: mat4x4<f32>, invProjectionMatrix: mat4x4<f32>, seed: ptr<function, u32>) -> Ray {
  let lookDirection = cameraWorld * vec4f(0.0, 0.0, -1.0, 0.0);
  let nearVector = invProjectionMatrix * vec4f(0.0, 0.0, -1.0, 1.0);
  let near = abs(nearVector.z / nearVector.w);

  var origin = cameraWorld * vec4f(0.0, 0.0, 0.0, 1.0);
  
  let randomOffset = randomF32(seed) * vec2f(0.5, 0.5);

  var direction = invProjectionMatrix * vec4f(coord.x, coord.y, 0.5, 1.0);
  direction /= direction.w;
  direction = cameraWorld * direction - origin;

  origin += vec4f(direction.xyz * near / dot(direction, lookDirection), 0);

  return Ray(
    origin.xyz,
    direction.xyz
  );
}

fn xorshift32(seed: ptr<function, u32>) -> u32 {
  var x = (*seed);
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  (*seed) = x;
  return x;
}

fn getPixelJitter(seed: ptr<function, u32>) -> vec2f {
  let jitterX = 0.5 * sampleTriangleFilter(randomF32(seed));
  let jitterY = 0.5 * sampleTriangleFilter(randomF32(seed));
  return vec2f(jitterX, jitterY);
}

@compute
@workgroup_size(${maxWorkgroupDimension}, ${maxWorkgroupDimension}, 1)
fn computeMain(@builtin(global_invocation_id) local_id: vec3<u32>) {
  // todo: use based on scene
  let invModelMatrix = identityMatrix();

  var seed = local_id.x + local_id.y * ${imageWidth};
  xorshift32(&seed);
  seed ^= uniformData.seedOffset;
  
  let i = f32(local_id.x);
  let j = f32(local_id.y);
  
  var pixelColor = vec3<f32>(0.0, 0.0, 0.0);
  
  let samplesPerPixel = i32(uniformData.samplesPerPixel);
  for (var sample = 0; sample < samplesPerPixel; sample += 1) {
    // CODE#ALIASING
    // anti-aliasing
    let pixel = vec2<f32>(i, j) + getPixelJitter(&seed);
    let ndc = -1.0 + 2.0*pixel / vec2<f32>(${imageWidth}, ${imageHeight});
    
    var ray = ndcToCameraRay(ndc, invModelMatrix * uniformData.cameraWorldMatrix, uniformData.invProjectionMatrix, &seed);
    ray.direction = normalize(ray.direction);

    pixelColor += rayColor(ray, &seed);
  }
  
  writeColor(pixelColor, i32(local_id.x), i32(local_id.y), samplesPerPixel);
}
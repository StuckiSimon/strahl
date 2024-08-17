alias Color = vec3f;

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
}

// Use due to 16 bytes alignment of vec3
struct IndicesPackage {
  x: i32,
  y: i32,
  z: i32,
}

// Use due to 16 bytes alignment of vec3
struct VertexPackage {
  x: f32,
  y: f32,
  z: f32,
}

// CODE#BUFFER-BINDINGS
@group(0) @binding(0) var<storage, read> positions: array<VertexPackage>;
// todo: Check when i16 is supported
@group(0) @binding(1) var<storage, read> indices: array<IndicesPackage>;

@group(0) @binding(2) var<storage, read> bounds: array<array<vec4f, 2>>;
@group(0) @binding(3) var<storage, read> contents: array<BinaryBvhNodeInfo>;

@group(0) @binding(4) var<storage, read> normals: array<VertexPackage>;

@group(0) @binding(5) var<storage, read> indirectIndices: array<u32>;

@group(0) @binding(6) var<storage, read> objectDefinitions: array<ObjectDefinition>;

@group(0) @binding(7) var<storage, read> materials: array<Material>;

@group(1) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;

@group(1) @binding(1) var readTexture: texture_storage_2d<rgba8unorm, read>;

@group(1) @binding(2) var<uniform> uniformData: UniformData;

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

fn ggxNDF(H: vec3f, alpha: vec2<f32>) -> f32 {
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
  return (word >> 22u) ^ word;
}

const range = 1.0 / f32(0xffffffffu);

// Generate a random float in the range [0, 1).
fn randomF32(seed: ptr<function, u32>) -> f32 {
  *seed = randomU32(*seed);
  return f32(*seed - 1u) * range;
}

const TRIANGLE_EPSILON = 1.0e-6;

// CODE#TRIANGLE-INTERSECTION
// Möller–Trumbore intersection algorithm without culling
fn triangleHit(triangle: Triangle, ray: Ray, rayT: Interval, hitRecord: ptr<function, HitRecord>) -> bool {
  let edge1 = triangle.u;
  let edge2 = triangle.v;
  let pvec = cross(ray.direction, edge2);
  let det = dot(edge1, pvec);
  // No hit if ray is parallel to the triangle (ray lies in plane of triangle)
  if (det > -TRIANGLE_EPSILON && det < TRIANGLE_EPSILON) {
    return false;
  }
  let invDet = 1.0 / det;
  let tvec = ray.origin - triangle.Q;
  let u = dot(tvec, pvec) * invDet;

  if (u < 0.0 || u > 1.0) {
    return false;
  }

  let qvec = cross(tvec, edge1);
  let v = dot(ray.direction, qvec) * invDet;

  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  let t = dot(edge2, qvec) * invDet;
  
  // check if the intersection point is within the ray's interval
  if (t < (rayT).min || t > (rayT).max) {
    return false;
  }

  (*hitRecord).t = t;
  (*hitRecord).point = rayAt(ray, t);
  (*hitRecord).normal = normalize(triangle.normal0 * (1.0 - u - v) + triangle.normal1 * u + triangle.normal2 * v);

  (*hitRecord).material = triangle.material;

  return true;
}

// CODE#BVH-TESTS
// Based on https://github.com/gkjohnson/three-mesh-bvh/blob/master/src/gpu/glsl/bvh_ray_functions.glsl.js
fn intersectsBounds(ray: Ray, boundsMin: vec3f, boundsMax: vec3f, dist: ptr<function, f32>) -> bool {
  let invDir = vec3f(1.0) / ray.direction;
  
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
  //  2 x x,y,z + unused alpha
  let boundaries = bounds[currNodeIndex];
  let boundsMin = boundaries[0];
  let boundsMax = boundaries[1];
  return intersectsBounds(ray, boundsMin.xyz, boundsMax.xyz, dist);
}

fn vertexPackageToVec3f(vp: VertexPackage) -> vec3f {
  return vec3f(vp.x, vp.y, vp.z);
}

fn intersectTriangles(offset: u32, count: u32, ray: Ray, rayT: Interval, hitRecord: ptr<function, HitRecord>) -> bool {
  var found = false;
  var localDist = hitRecord.t;
  let l = offset + count;
  
  for (var i = offset; i < l; i += 1) {
    let indAccess = indirectIndices[i];
    let indicesPackage = indices[indAccess];
    let v1Index = indicesPackage.x;
    let v2Index = indicesPackage.y;
    let v3Index = indicesPackage.z;
    
    let x = vertexPackageToVec3f(positions[v1Index]);
    let y = vertexPackageToVec3f(positions[v2Index]);
    let z = vertexPackageToVec3f(positions[v3Index]);
    
    let Q = x;
    let u = y - x;
    let v = z - x;
    
    let vIndexOffset = indAccess * 3;
    var matchingObjectDefinition: ObjectDefinition = objectDefinitions[0];
    for (var j = 0; j < uniformData.objectDefinitionLength ; j++) {
      let objectDefinition = objectDefinitions[j];
      if (objectDefinition.start <= vIndexOffset && objectDefinition.start + objectDefinition.count > vIndexOffset) {
        matchingObjectDefinition = objectDefinition;
        break;
      }
    }
    let materialDefinition = matchingObjectDefinition.material;

    let normalX = vertexPackageToVec3f(normals[v1Index]);
    let normalY = vertexPackageToVec3f(normals[v2Index]);
    let normalZ = vertexPackageToVec3f(normals[v3Index]);
    
    let triangle = Triangle(Q, u, v, materialDefinition, normalX, normalY, normalZ);

    var tmpRecord: HitRecord;
    if (triangleHit(triangle, ray, Interval(rayT.min, localDist), &tmpRecord)) {
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

fn hittableListHit(ray: Ray, rayT: Interval, hitRecord: ptr<function, HitRecord>) -> bool {
  var tempRecord: HitRecord;
  var hitAnything = false;
  var closestSoFar = rayT.max;

  // Inspired by https://github.com/gkjohnson/three-mesh-bvh/blob/master/src/gpu/glsl/bvh_ray_functions.glsl.js
  
  // BVH Intersection Detection
  var sPtr = 0;
  var stack: array<u32, ${maxBvhStackDepth}> = array<u32, ${maxBvhStackDepth}>();
  stack[sPtr] = 0u;

  while (sPtr > -1 && sPtr < ${maxBvhStackDepth}) {
    let currNodeIndex = stack[sPtr];
    sPtr -= 1;

    var boundsHitDistance: f32;
    
    if (!intersectsBVHNodeBounds(ray, currNodeIndex, &boundsHitDistance) || boundsHitDistance > closestSoFar) {
      continue;
    }

    let boundsInfo = contents[currNodeIndex];
    let boundsInfoX = boundsInfo.x;
    let boundsInfoY = boundsInfo.y;

    // CODE#BVH-NODE-ACCESS
    let isLeaf = (boundsInfoX & 0xffff0000u) == 0xffff0000u;

    if (isLeaf) {
      let count = boundsInfoX & 0x0000ffffu;
      let offset = boundsInfoY;

      let found2 = intersectTriangles(
        offset,
        count,
        ray,
        rayT,
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

  let D = ggxNDF(mR, alpha);
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

fn fresnelDielectricPolarizations(mui: f32, etaTi: f32) -> vec2f {
  let mut2 = sqr(etaTi) - (1.0 - sqr(mui));
  if (mut2 <= 0.0) {
    return vec2f(1.0);
  }

  let mut1 = sqrt(mut2) / etaTi;
  let rs = (mui - etaTi*mut1) / (mui + etaTi*mut1);
  let rp = (mut1 - etaTi*mui) / (mut1 + etaTi*mui);
  return vec2f(rs, rp);
}

fn fresnelDielectricReflectance(mui: f32, etaTi: f32) -> f32 {
  let r = fresnelDielectricPolarizations(mui, etaTi);
  return 0.5 * dot(r, r);
}

fn specularBrdfSample(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, seed: ptr<function, u32>, woutputL: ptr<function, vec3f>, pdfWoutputL: ptr<function, f32>) -> vec3f {
  let beamOutgoingL = winputL;
  let externalReflection = beamOutgoingL.z > 0.0;

  let etaIe = specularIorRatio(material);
  let etaTiRefl = select(1.0/etaIe, etaIe, externalReflection);
  if (abs(etaTiRefl - 1.0) < IOR_EPSILON) {
    // (*pdfWoutputL) = PDF_EPSILON; // todo: reset?
    return vec3f(0.0);
  }

  let tint = material.specularColor;

  let alpha = specularNDFRoughness(material);

  let rotation = getLocalFrameRotation(2*PI*material.specularRotation);
  let winputR = localToRotated(winputL, rotation);

  var mR: vec3f;
  if (winputR.z > 0.0) {
    mR = ggxNDFSample(winputR, alpha, seed);
  } else {
    var winputRReflected = winputR;
    winputRReflected.z = -winputRReflected.z;
    mR = ggxNDFSample(winputRReflected, alpha, seed);
    mR.z = -mR.z;
  }

  var woutputR = -winputR + 2.0*dot(winputR, mR)*mR;
  if (winputR.z * woutputR.z < 0.0) {
    (*pdfWoutputL) = 1.0;
    return vec3f(0.0);
  }

  (*woutputL) = rotatedToLocal(woutputR, rotation);

  let D = ggxNDFEval(mR, alpha);
  let DV = D * ggxG1(winputR, alpha) * abs(dot(winputR, mR)) / max(DENOM_TOLERANCE, abs(winputR.z));

  let dwhDwo = 1.0 / max(abs(4.0*dot(winputR, mR)), DENOM_TOLERANCE);
  (*pdfWoutputL) = DV * dwhDwo;

  let G2 = ggxG2(winputR, woutputR, alpha);

  // todo: coat workflow
  let F = vec3f(fresnelDielectricReflectance(abs(dot(winputR, mR)), etaTiRefl));
  
  let f = F * D * G2 / max(4.0 * abs(woutputL.z) * abs(winputL.z), DENOM_TOLERANCE);

  return f * tint;
}

fn specularBrdfEvaluate(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, woutputL: vec3f, pdfWoutputL: ptr<function, f32>) -> vec3f {
  let transmitted = woutputL.z * winputL.z < 0.0;
  if (transmitted) {
    // (*pdfWoutputL) = PDF_EPSILON; todo: reset?
    return vec3f(0.0);
  }

  let beamOutgoingL = winputL;
  let externalReflection = beamOutgoingL.z > 0.0;

  let etaIe = specularIorRatio(material);
  let etaTiRefl = select(1.0/etaIe, etaIe, externalReflection);
  if (abs(etaTiRefl - 1.0) < IOR_EPSILON) {
    return vec3f(0.0);
  }

  let tint = material.specularColor;

  let alpha = specularNDFRoughness(material);

  let rotation = getLocalFrameRotation(2*PI*material.specularRotation);
  let winputR = localToRotated(winputL, rotation);
  let woutputR = localToRotated(woutputL, rotation);

  let mR = normalize(woutputR + winputR);

  if (dot(mR, winputR) * winputR.z < 0.0 || dot(mR, woutputR) * woutputR.z < 0.0) {
    return vec3f(0.0);
  }

  let D = ggxNDFEval(mR, alpha);
  let DV = D * ggxG1(winputR, alpha) * max(0.0, dot(winputR, mR)) / max(DENOM_TOLERANCE, winputR.z);

  let dwhDwo = 1.0 / max(abs(4.0*dot(winputR, mR)), DENOM_TOLERANCE);
  (*pdfWoutputL) = DV * dwhDwo;

  let G2 = ggxG2(winputR, woutputR, alpha);

  // todo: coat workflow
  let F = vec3f(fresnelDielectricReflectance(abs(dot(winputR, mR)), etaTiRefl));

  let f = F * D * G2 / max(4.0 * abs(woutputL.z) * abs(winputL.z), DENOM_TOLERANCE);
  return f * tint;
}

fn etaS(material: Material) -> f32 {
  const ambientIor = 1.0;
  let coatIorAverage = mix(ambientIor, material.coatIor, material.coatWeight);
  let etaS = material.specularIor / coatIorAverage;
  return etaS;
}

fn fresnelReflNormalIncidence(material: Material) -> f32 {
  let etaS = etaS(material);
  let Fs = sqr((etaS - 1.0)/(etaS + 1.0));
  return Fs;
}

fn specularIorRatio(material: Material) -> f32 {
  let Fs = fresnelReflNormalIncidence(material);
  let xiS = clamp(material.specularWeight, 0.0, 1.0/max(Fs, DENOM_TOLERANCE));
  let etaS = etaS(material);
  let temp = min(1.0, sign(etaS - 1.0) * sqrt(xiS * Fs));
  let etaSPrime = (1.0 + temp) / max(1.0 - temp, DENOM_TOLERANCE);
  return etaSPrime;
}

fn specularBrdfAlbedo(material: Material, pW: vec3f, basis: Basis, winputL: vec3f, seed: ptr<function, u32>) -> vec3f {
  let etaIe = specularIorRatio(material);
  if (abs(etaIe - 1.0) < IOR_EPSILON) {
    return vec3f(0.0);
  }

  const samples = 1;
  var albedo = vec3f(0.0);
  for (var n = 0; n < samples; n += 1) {
    var woutputL: vec3f;
    var pdfWoutputL: f32;
    var f = specularBrdfSample(material, pW, basis, winputL, seed, &woutputL, &pdfWoutputL);
    if (length(f) > RADIANCE_EPSILON) {
      albedo += f * abs(woutputL.z) / max(DENOM_TOLERANCE, pdfWoutputL);
    }
  }
  albedo /= f32(samples);

  return albedo;
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
  albedos.m[ID_SPEC_BRDF] = select(vec3f(0.0), specularBrdfAlbedo(material, pW, basis, winputL, seed), !fullyMetallic);
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
  for (var lobeId = 0; lobeId < NUM_LOBES; lobeId += 1) {
    probs.m[lobeId] = length(weights.m[lobeId] * albedos.m[lobeId]);
    Wtotal += probs.m[lobeId];
  }
  Wtotal = max(DENOM_TOLERANCE, Wtotal);
  for (var lobeId = 0; lobeId < NUM_LOBES; lobeId += 1) {
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
const IOR_EPSILON = 1.0e-5;
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

fn openpbrBsdfEvaluateLobes(pW: vec3f, basis: Basis, material: Material, winputL: vec3f, woutputL: vec3f, skipLobeId: i32, lobeData: LobeData, pdfs: ptr<function, LobePDFs>) -> vec3f {
  var f = vec3f(0.0);
  if (skipLobeId != ID_FUZZ_BRDF && lobeData.probs.m[ID_FUZZ_BRDF] > 0.0) {
    f += vec3f(0.0);
  }
  if (skipLobeId != ID_COAT_BRDF && lobeData.probs.m[ID_COAT_BRDF] > 0.0) {
    f += lobeData.weights.m[ID_COAT_BRDF] * brdfEvaluatePlaceholder();
  }
  if (skipLobeId != ID_META_BRDF && lobeData.probs.m[ID_META_BRDF] > 0.0) {
    f += metalBrdfEvaluate(pW, basis, winputL, woutputL, material, &pdfs.m[ID_META_BRDF]);
  }
  if (skipLobeId != ID_SPEC_BRDF && lobeData.probs.m[ID_SPEC_BRDF] > 0.0) {
    f += lobeData.weights.m[ID_SPEC_BRDF] * specularBrdfEvaluate(material, pW, basis, winputL, woutputL, &pdfs.m[ID_SPEC_BRDF]);
  }
  if (skipLobeId != ID_DIFF_BRDF && lobeData.probs.m[ID_DIFF_BRDF] > 0.0) {
    f += lobeData.weights.m[ID_DIFF_BRDF] * diffuseBrdfEvaluate(material, pW, basis, winputL, woutputL, &pdfs.m[ID_DIFF_BRDF]);
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
      else if (lobeId == ID_SPEC_BRDF) {
        fLobe = specularBrdfSample(material, pW, basis, winputL, seed, woutputL, &pdfLobe);
      }
      else if (lobeId == ID_SPEC_BTDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_SSSC_BTDF) { fLobe = brdfSamplePlaceholder(); }
      else if (lobeId == ID_DIFF_BRDF) {
        fLobe = diffuseBrdfSample(material, pW, basis, winputL, woutputL, &pdfLobe, seed);
        }
      else { break; }

      var pdfs: LobePDFs;
      var skipLobeId = lobeId;
      var f = openpbrBsdfEvaluateLobes(pW, basis, material, winputL, *woutputL, skipLobeId, lobeData, &pdfs);
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

fn evaluateBsdf(pW: vec3f, basis: Basis, winputL: vec3f, woutputL: vec3f, lobeData: LobeData, material: Material, pdfWoutputL: ptr<function, f32>) -> vec3f {
  var pdfs: LobePDFs;
  let f = openpbrBsdfEvaluateLobes(pW, basis, material, winputL, woutputL, -1, lobeData, &pdfs);
  (*pdfWoutputL) = openpbrBsdfTotalPdf(pdfs, lobeData);
  
  return f;
}

fn evaluateEdf(material: Material) -> vec3f {
  return material.emissionColor * material.emissionLuminance;
}

fn sunSample(basis: Basis, sunBasis: Basis, woutputL: ptr<function, vec3f>, woutputW: ptr<function, vec3f>, pdfDir: ptr<function, f32>, seed: ptr<function, u32>) -> vec3f {
  let thetaMax = uniformData.sunAngularSize * PI/180.0;
  let theta = thetaMax * sqrt(randomF32(seed));
  let cosTheta = cos(theta);
  let sinTheta = sqrt(max(0, 1.0-cosTheta*cosTheta));
  let phi = 2.0 * PI * randomF32(seed);
  let cosPhi = cos(phi);
  let sinPhi = sin(phi);
  let x = sinTheta * cosPhi;
  let y = sinTheta * sinPhi;
  let z = cosTheta;
  let solidAngle = 2.0 * PI * (1.0 - cos(thetaMax));
  *pdfDir = 1.0 / solidAngle;
  *woutputW = localToWorld(vec3f(x, y, z), sunBasis);
  *woutputL = worldToLocal(*woutputW, basis);
  return uniformData.sunPower * uniformData.sunColor;
}

fn skySample(basis: Basis, woutputL: ptr<function, vec3f>, woutputW: ptr<function, vec3f>, pdfDir: ptr<function, f32>, seed: ptr<function, u32>) -> vec3f {
  *woutputL = sampleHemisphereCosineWeighted(pdfDir, seed);
  *woutputW = localToWorld(*woutputL, basis);
  return skyRadiance();
}

fn getDirectLighting(pW: vec3f, basis: Basis, sunBasis: Basis, shadowL: ptr<function, vec3f>, shadowW: ptr<function, vec3f>, lightPdf: ptr<function, f32>, seed: ptr<function, u32>) -> vec3f {
  var Li: vec3f;

  let wSun = sunTotalPower();
  let wSky = skyTotalPower();
  let pSun = wSun / (wSun + wSky);
  let pSky = max(0.0, 1.0 - pSun);
  var pdfSun: f32;
  var pdfSky: f32;
  let r = randomF32(seed);
  if (r < pSun) {
    Li = sunSample(basis, sunBasis, shadowL, shadowW, &pdfSun, seed);
    Li += skyRadiance();
    pdfSky = skyPdf(*shadowL, *shadowW);
  } else {
    Li = skySample(basis, shadowL, shadowW, &pdfSky, seed);
    Li += sunRadiance(*shadowW);
    pdfSun = sunPdf(*shadowL, *shadowW);
  }
  *lightPdf = pSun * pdfSun + pSky * pdfSky;

  if (shadowL.z < 0) {
    return vec3f(0);
  }
  if (maxVec3(Li) < RADIANCE_EPSILON) {
    return vec3f(0);
  }

  let occluded = isOccluded(Ray(pW, *shadowW), TRIANGLE_MAX_DISTANCE_THRESHOLD);
  let visibility = select(1.0, 0.0, occluded);

  return visibility * Li;
}

fn isOccluded(ray: Ray, maxDistance: f32) -> bool {
  var hitRecord = HitRecord();
  hitRecord.t = maxDistance;
  return hittableListHit(ray, Interval(TRIANGLE_MIN_DISTANCE_THRESHOLD, maxDistance), &hitRecord);
}

const TRIANGLE_MIN_DISTANCE_THRESHOLD = 0.0005;
const TRIANGLE_MAX_DISTANCE_THRESHOLD = 10e37f;

fn rayColor(cameraRay: Ray, seed: ptr<function, u32>, sunBasis: Basis) -> vec4f {
  var hitRecord: HitRecord;
  var ray = cameraRay;

  var throughput = vec3f(1.0);
  var L = vec3f(0.0);
  var bsdfPdfContinuation = 1.0;

  var dW = ray.direction;
  var pW = ray.origin;

  var basis: Basis;

  var inDielectric = false;

  for (var i = 0; i < uniformData.maxRayDepth; i += 1) {
    // todo: handle setting t nicely
    hitRecord.t = TRIANGLE_MAX_DISTANCE_THRESHOLD;
    let hit = hittableListHit(ray, Interval(TRIANGLE_MIN_DISTANCE_THRESHOLD, TRIANGLE_MAX_DISTANCE_THRESHOLD), &hitRecord);

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
          return vec4f(uniformData.clearColor, 0.0);
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

    L += throughput * evaluateEdf(material);

    pW += NgW * sign(dot(dW, NgW)) * RAY_OFFSET;

    ray = Ray(pW, dW);

    var transmitted = dot(winputW, NgW) * dot(dW, NgW) < 0.0;
    if (transmitted) {
      inDielectric = !inDielectric;
    }

    if (!inDielectric && !transmitted) {
      var shadowL: vec3f;
      var shadowW: vec3f;
      var lightPdf: f32;
      let Li = getDirectLighting(pW, basis, sunBasis, &shadowL, &shadowW, &lightPdf, seed);
      if (maxVec3(Li) > RADIANCE_EPSILON) {
        var bsdfPdfShadow = PDF_EPSILON;
        let fShadow = evaluateBsdf(pW, basis, winputL, shadowL, lobeData, material, &bsdfPdfShadow);
        let misWeightLight = powerHeuristic(lightPdf, bsdfPdfShadow);
        L += throughput * misWeightLight * fShadow * abs(dot(shadowW, basis.nW)) * Li / max(PDF_EPSILON, lightPdf);
      }
    }

    throughput *= surfaceThroughput;

    // CODE#RUSSIAN-ROULETTE
    // Russian Roulette
    if (maxVec3(throughput) < 1.0 && i > 1) {
      let q = max(0.0, 1.0 - maxVec3(throughput));
      if (randomF32(seed) < q) {
        break;
      }
      throughput /= 1.0 - q;
    }
  }

  return vec4f(L, 1.0);
}

fn writeColor(pixelColor: vec4f, x: i32, y: i32, samples: i32) {
  let previousColor = textureLoad(readTexture, vec2<i32>(x, y));
  let previousColorAdjusted = previousColor * f32(uniformData.priorSamples);
  let samplesPerPixel = uniformData.samplesPerPixel;
  let scale = 1.0 / f32(uniformData.priorSamples + samplesPerPixel);
  let adjustedColor = (pixelColor + previousColorAdjusted) * scale;
  textureStore(texture, vec2<i32>(x, y), adjustedColor);
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

fn getPixelJitter(seed: ptr<function, u32>) -> vec2f {
  let jitterX = 0.5 * sampleTriangleFilter(randomF32(seed));
  let jitterY = 0.5 * sampleTriangleFilter(randomF32(seed));
  return vec2f(jitterX, jitterY);
}

@compute
@workgroup_size(${maxWorkgroupDimension}, ${maxWorkgroupDimension}, 1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  var seed = globalId.x + globalId.y * ${imageWidth};
  seed ^= uniformData.seedOffset;

  let pixelOrigin = vec2f(f32(globalId.x), f32(globalId.y));
  
  var pixelColor = vec4f(0.0);

  // todo: consider not re-creating the basis every time
  var sunBasis = makeBasis(uniformData.sunDirection);
  
  let samplesPerPixel = i32(uniformData.samplesPerPixel);
  for (var sample = 0; sample < samplesPerPixel; sample += 1) {
    // CODE#ALIASING
    // anti-aliasing
    let pixel = pixelOrigin + getPixelJitter(&seed);
    let ndc = -1.0 + 2.0*pixel / vec2<f32>(${imageWidth}, ${imageHeight});
    
    var ray = ndcToCameraRay(ndc, uniformData.invModelMatrix * uniformData.cameraWorldMatrix, uniformData.invProjectionMatrix, &seed);
    ray.direction = normalize(ray.direction);

    pixelColor += rayColor(ray, &seed, sunBasis);
  }

  
  writeColor(pixelColor, i32(globalId.x), i32(globalId.y), samplesPerPixel);
}

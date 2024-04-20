@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> bvhNodes: array<BvhNode>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;
// todo: Check when i16 is supported
@group(0) @binding(3) var<storage, read_write> indices: array<i32>;

@group(0) @binding(4) var<storage, read_write> bounds: array<f32>;
@group(0) @binding(5) var<storage, read_write> contents: array<u32>;

// todo: This should not be hardcoded
const indicesLength = 12636;

const focalLength = 1.0;
const viewportHeight = 4.0;
const viewportWidth = 4.0;
const cameraCenter = vec3<f32>(0.0, 0.0, 0.0);

const viewportU = vec3<f32>(viewportWidth, 0.0, 0.0);
const viewportV = vec3<f32>(0.0, -viewportHeight, 0.0);

const pixelDeltaU = viewportU / ${imageWidth};
const pixelDeltaV = viewportV / ${imageHeight};

const viewportUpperLeft = cameraCenter - vec3<f32>(0, 0, focalLength) - viewportU / 2.0 - viewportV / 2.0;
const pixel00Loc = viewportUpperLeft + 0.5 * (pixelDeltaU + pixelDeltaV);

const samplesPerPixel = 1;
const maxDepth = 2;

const MINIMUM_FLOAT_EPSILON = 1e-8;
const PI = 3.1415926535897932;
const PI_INVERSE = 1.0 / PI;

struct Ray {
  origin: vec3<f32>,
  direction: vec3<f32>,
};

struct MaterialDefinition {
  index: i32,
}

struct HitRecord {
  point: vec3<f32>,
  normal: vec3<f32>,
  t: f32,
  frontFace: bool,
  material: MaterialDefinition,
}

alias Color = vec3<f32>;

struct Material {
  baseWeight: f32,
  baseColor: Color,
  baseRoughness: f32,
  baseMetalness: f32,
  specularWeight: f32,
  specularColor: Color,
  specularRoughness: f32,
  specularAnisotropy: f32,
  specularRotation: f32,
  coatWeight: f32,
  coatRoughness: f32,
  emissionLuminance: f32,
  emissionColor: Color,
  thinFilmThickness: f32,
  thinFilmIOR: f32,
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

fn interval(min: f32, max: f32) -> Interval {
  return Interval(min, max);
}

fn intervalContains(interval: Interval, value: f32) -> bool {
  return interval.min <= value && value <= interval.max;
}

fn intervalSurrounds(interval: Interval, value: f32) -> bool {
  return interval.min < value && value < interval.max;
}

fn intervalClamp(interval: Interval, value: f32) -> f32 {
  return clamp(value, interval.min, interval.max);
}

struct Aabb {
  intervalX: Interval,
  intervalY: Interval,
  intervalZ: Interval,
}

fn aabbAxis(aabb: Aabb, axis: i32) -> Interval {
  if (axis == 0) {
    return aabb.intervalX;
  } else if (axis == 1) {
    return aabb.intervalY;
  } else {
    return aabb.intervalZ;
  }
}

fn aabbHit(aabb: Aabb, r: Ray, rayT: ptr<function, Interval>) -> bool {
  for (var a = 0; a < 3; a += 1) {
    let t0 = min((aabbAxis(aabb, a).min - r.origin[a]) / r.direction[a],
                 (aabbAxis(aabb, a).max - r.origin[a]) / r.direction[a]);
    let t1 = max((aabbAxis(aabb, a).min - r.origin[a]) / r.direction[a],
                  (aabbAxis(aabb, a).max - r.origin[a]) / r.direction[a]);
    (*rayT).min = max((*rayT).min, t0);
    (*rayT).max = min((*rayT).max, t1);
    if ((*rayT).max <= (*rayT).min) {
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
  let epsilon = vec3f(MINIMUM_FLOAT_EPSILON);
  return any(abs(v) < epsilon);
}

fn identical(v1: vec3f, v2: vec3f) -> bool {
  return all(v1 == v2);
}

fn forwardFacingNormal(v1: vec3f, v2: vec3f) -> vec3f {
  return select(v1, -v1, dot(v1, v2) < 0.0);
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

// https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
// Appendix B.2 Equation 13
fn ggxNDF(H: vec3f, alpha: vec2<f32>) -> f32 {
  let He = H.xy / alpha;
  let denom = dot(He, He) + (H.z*H.z);
  return 1.0 / (PI * alpha.x * alpha.y * denom * denom);
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

fn computeFresnel(cosTheta: f32, fd: FresnelData) -> vec3f {
  // todo: implement other models (dielectric, conductor, airy)
  if (fd.model == FRESNEL_MODEL_SCHLICK) {
    return fresnelSchlick(cosTheta, fd.F0, fd.F90, fd.exponent);
  }
  
  return vec3f(0.0);
}

fn generalizedSchlickBsdfReflection(L: vec3f, V: vec3f, P: vec3f, occlusion: f32, weight: f32, color0: Color, color90: Color, exponent: f32, roughness: vec2<f32>, N_in: vec3f, X_in: vec3f, distribution: i32, scatterMode: i32, bsdfThickness: f32, bsdfIor: f32) -> BsdfResponse {

  if (weight < MINIMUM_FLOAT_EPSILON) {
    return BsdfResponse(
      vec3f(0.0),
      vec3f(0.0),
      0.0,
      0.0
    );
  }

  let N = forwardFacingNormal(N_in, V);
  
  let X = normalize(X_in - dot(X_in, N) * N);
  let Y = cross(N, X);
  let H = normalize(L + V);

  let NdotL = clamp(dot(N, L), MINIMUM_FLOAT_EPSILON, 1.0);
  let NdotV = clamp(dot(N, V), MINIMUM_FLOAT_EPSILON, 1.0);
  let VdotH = clamp(dot(V, H), MINIMUM_FLOAT_EPSILON, 1.0);

  let safeAlpha = clamp(roughness, vec2(MINIMUM_FLOAT_EPSILON, MINIMUM_FLOAT_EPSILON), vec2(1.0, 1.0));
  let avgAlpha = averageAlpha(safeAlpha);
  let Ht = vec3f(dot(H, X), dot(H, Y), dot(H, N));

  var fd: FresnelData;
  let safeColor0 = max(color0, vec3f(0, 0, 0));
  let safeColor90 = max(color90, vec3f(0, 0, 0));
  if (bsdfThickness > 0.0) {
    // todo: implement fresnelSchlickAiry
  } else {
    fd = initFresnelSchlick(safeColor0, safeColor90, exponent);
  }
  let F = computeFresnel(VdotH, fd);
  let D = ggxNDF(Ht, safeAlpha);
  let G = ggxSmithG2(NdotL, NdotV, avgAlpha);

  let comp = ggxEnergyCompensation(NdotV, avgAlpha, F);
  let dirAlbedo = ggxDirAlbedo(NdotV, avgAlpha, safeColor0, safeColor90);
  let avgDirAlbedo = dot(dirAlbedo, vec3f(1.0 / 3.0));
  let bsdfThroughput = vec3f(1.0 - avgDirAlbedo * weight);

  let bsdfResponse = D * F * G * comp * occlusion * weight / (4.0 * NdotV);

  return BsdfResponse(
    // ensure to not return infinitesimal values
    max(bsdfResponse, vec3f(MINIMUM_FLOAT_EPSILON, MINIMUM_FLOAT_EPSILON, MINIMUM_FLOAT_EPSILON)),
    bsdfThroughput,
    bsdfThickness,
    bsdfIor
  );
}

fn openPbrAnisotropy(roughness: f32, anisotropy: f32) -> vec2f {
  let roughness2 = roughness * roughness;
  let anisotropyInverted = 1.0 - anisotropy;
  let anisotropyInverted2 = anisotropyInverted * anisotropyInverted;
  let denom = anisotropyInverted2 + 1;
  let fraction = 2 / denom;
  let sqrt = sqrt(fraction);
  let alphaX = roughness2 * sqrt;
  let alphaY = anisotropyInverted * alphaX;
  return vec2f(alphaX, alphaY);
}

fn rotationMatrix(originalAxis: vec3f, angle: f32) -> mat4x4<f32> {
  let axis = normalize(originalAxis);
  let s = sin(angle);
  let c = cos(angle);
  let oc = 1.0 - c;

  return mat4x4<f32>(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                     oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                     oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                     0.0,                                0.0,                                0.0,                                1.0);
}

fn rotateVec3(in: vec3f, amount: f32, axis: vec3f) -> vec3f {
  let rotationRadians = radians(amount);
  let rotationMatrix = rotationMatrix(axis, rotationRadians);
  return (rotationMatrix * vec4f(in, 1.0)).xyz;
}

fn renderMaterial(material: Material, hitRecord: HitRecord, attenuation: ptr<function, Color>, emissionColor: ptr<function, Color>, scattered: ptr<function, Ray>, seed: ptr<function, u32>) -> bool {
  let incomingRay = scattered;
  
  // Inverse-square law
  let intensityFactor = material.emissionLuminance / pow(hitRecord.t, 2);
  (*emissionColor) = min(material.emissionColor * intensityFactor, Color(1.0, 1.0, 1.0));

  // todo: optimize?
  if (identical(material.baseColor, Color(0.0, 0.0, 0.0))) {
    return false;
  }
  
  let randomScatter = hitRecord.normal + normalize(randInUnitSphere(seed));
  let specularScatter = normalize(reflect((*incomingRay).direction, hitRecord.normal));

  let scatterMix = mix(specularScatter, randomScatter, material.baseRoughness);

  var scatterDirection = scatterMix;

  // Catch degenerate scatter direction
  if (nearZero(scatterDirection)) {
    scatterDirection = hitRecord.normal;
  }

  (*scattered) = Ray(hitRecord.point, scatterDirection);

  // light direction
  let L = normalize(scatterDirection);
  // view direction
  let V = normalize(-(*incomingRay).direction);
  // hit point
  let P = hitRecord.point;

  let occlusion = 1.0;

  let coatAffectedRoughnessFg = 1.0;
  let coatAffectRoughnessMultiply2 = material.coatWeight * material.coatRoughness;
  let coatAffectedRoughness = mix(material.specularRoughness, coatAffectedRoughnessFg, coatAffectRoughnessMultiply2);
  let mainRoughness = openPbrAnisotropy(coatAffectedRoughness, material.specularAnisotropy);
  let tangentRotateDegree = material.specularRotation * 360.0;

  let geompropNworld = hitRecord.normal;
  // todo: check tangent vector (ensure to use normalized vector)
  let geompropTworld = vec3<f32>(1.0, 0.0, 0.0);
  let tangentRotate = rotateVec3(geompropTworld, tangentRotateDegree, geompropNworld);
  let tangentRotateNormalized = normalize(tangentRotate);

  let mainTangent = select(geompropTworld, tangentRotateNormalized, material.specularAnisotropy > 0.0);
  let metalBsdfWeight = 1.0;
  let metalBsdfExponent = 5.0;
  let metalReflectivity = material.baseColor * material.baseWeight;
  let metalEdgeColor = material.specularColor * material.specularWeight;
  let metalBsdfOut = generalizedSchlickBsdfReflection(
    L,
    V,
    P,
    occlusion,
    metalBsdfWeight,
    metalReflectivity,
    metalEdgeColor,
    metalBsdfExponent,
    mainRoughness,
    geompropNworld,
    mainTangent,
    // metalBsdfDistribution
    0,
    // metalBsdfScatterMode
    0,
    material.thinFilmThickness,
    material.thinFilmIOR
  );

  // Oren Nayar Diffuse BSDF Reflection based on MaterialX GLSL implementation
  // todo: also consider BsdfResponse (not only BsdfResponse.response)
  
  let normal = forwardFacingNormal(geompropNworld, V);
  let NdotL = clamp(dot(normal, L), MINIMUM_FLOAT_EPSILON, 1.0);

  var bsdfResponse = material.baseColor * occlusion * material.baseWeight * PI_INVERSE;

  if (material.baseRoughness > 0.0) {
    bsdfResponse *= orenNayarDiffuse(L, V, normal, NdotL, material.baseRoughness);
  }
  
  // opaque_base_out (ss not implemented atm)
  let opaqueBaseOut = bsdfResponse;

  let metalOpaqueLayerMix = mix(opaqueBaseOut, metalBsdfOut.response, material.baseMetalness);
  (*attenuation) = metalOpaqueLayerMix;

  return true;
}

const materials: array<Material, 4> = array<Material, 4>(
  // base materials
  Material(0.8, Color(0.5, 1.0, 0.0), 0.0, 1.0, 1.0, Color(0.5, 0.5, 1.0), 1.0, 0.5, 0.5, 0, 0, 0, Color(0.0, 0.0, 0.0), 0, 1.5),
  Material(1.0, Color(1.0, 0.5, 0.2), 0.0, 0.0, 1.0, Color(0.1, 0.1, 1.0), 1.0, 0.5, 0.5, 0, 0, 0, Color(0.0, 0.0, 0.0), 0, 1.5),
  // lights
  Material(0.0, Color(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, Color(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0, 10.0, Color(1.0, 0.5, 1.0), 0, 1.5),
  Material(0.0, Color(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, Color(0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0, 10.0, Color(1.0, 1.0, 1.0), 0, 1.5)
);

const defaultMaterial = MaterialDefinition(0);
const defaultDiffuseLightMaterial = MaterialDefinition(2);

const defaultNormal = vec3<f32>(0.0, 1.0, 0.0);

const TRIANGLE_COUNT = 6;
// Triangles are encoded as first being the lower point, then the two edges
const triangles: array<Triangle, TRIANGLE_COUNT> = array<Triangle, TRIANGLE_COUNT>(
  // wall facing camera
  Triangle(vec3<f32>(-3, 0, -3), vec3<f32>(4, 0, 0), vec3<f32>(0, 4, 0), defaultMaterial, defaultNormal,defaultNormal,defaultNormal),
  Triangle(vec3<f32>(-3, 4, -3), vec3<f32>(4, -4, 0), vec3<f32>(4, 0, 0), MaterialDefinition(1), defaultNormal,defaultNormal,defaultNormal),
  // ground floor below the wall
  Triangle(vec3<f32>(-3, 0, -3), vec3<f32>(4, 0, 0), vec3<f32>(0, -2, 1.5), MaterialDefinition(1), defaultNormal,defaultNormal,defaultNormal),
  Triangle(vec3<f32>(1, 0, -3), vec3<f32>(0, -2, 1.5), vec3<f32>(-4, -2, 1.5), defaultMaterial, defaultNormal,defaultNormal,defaultNormal),
  // Light source from the right
  Triangle(vec3<f32>(2, 5, -6), vec3<f32>(0.1, -20, 0), vec3<f32>(0, 0, 10), defaultDiffuseLightMaterial, defaultNormal,defaultNormal,defaultNormal),
  // Light from the left
  Triangle(vec3<f32>(-3, 5, -6), vec3<f32>(0.0, -20, 0), vec3<f32>(0, 0, 10), MaterialDefinition(3), defaultNormal,defaultNormal,defaultNormal),
);

fn bvhNodeHit(bvh: BvhNode, r: Ray, rayT: ptr<function, Interval>) -> bool {
  return aabbHit(bvh.boundingBox, r, rayT);
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
fn randomI32(seed: ptr<function, u32>) -> i32 {
  let oldstate = *seed;
  *seed = *seed * PCG_MULTIPLIER + PCG_INC;
  let word = ((oldstate >> ((oldstate >> 28u) + 4u)) ^ oldstate) * 277803737u;
  return i32((word >> 22u) ^ word);
}

fn randomF32(seed: ptr<function, u32>) -> f32 {
  let val = randomI32(seed);
  return f32(val) / f32(0xffffffffu);
}

fn randomF32InRange(min: f32, max: f32, seed: ptr<function, u32>) -> f32 {
  return (randomF32(seed) * (max - min)) + min;
}

fn randVec3(seed: ptr<function, u32>) -> vec3<f32> {
  return vec3<f32>(randomF32(seed), randomF32(seed), randomF32(seed));
}

fn randVec3InRange(min: f32, max: f32, seed: ptr<function, u32>) -> vec3<f32> {
  return vec3<f32>(
    randomF32InRange(min, max, seed),
    randomF32InRange(min, max, seed),
    randomF32InRange(min, max, seed),
  );
}

fn randInUnitSphere(seed: ptr<function, u32>) -> vec3<f32> {
  while (true) {
    let p = randVec3(seed);
    if (lengthSquared(p) >= 1.0) {
      continue;
    }
    return p;
  }
  // Should never reach here
  return vec3<f32>(0.0, 0.0, 0.0);
}

fn triangleBoundingBox(triangle: Triangle) -> Aabb {
  let min = min(triangle.Q, min(triangle.Q + triangle.u, triangle.Q + triangle.v));
  let max = max(triangle.Q, max(triangle.Q + triangle.u, triangle.Q + triangle.v));
  return Aabb(interval(min.x, max.x), interval(min.y, max.y), interval(min.z, max.z));
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

  return true;
}

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

  // todo: use bvh
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
  }

  return hitAnything;
}

struct BouncingInfo {
  attenuation: Color,
  emission: Color,
}

fn rayColor(ray: Ray, seed: ptr<function, u32>) -> vec3<f32> {
  var hitRecord: HitRecord;
  var localRay = ray;

  var colorStack: array<BouncingInfo, maxDepth>;
  var colorStackIdx = -1;

  for (var i = 0; i < maxDepth; i += 1) {
    if (hittableListHit(localRay, Interval(0.001, 0xfffffffffffffff), &hitRecord)) {
      var attenuation: Color;
      
      var emissionColor = Color(0,0,0);
      
      let material = hitRecord.material;
      let scattered = renderMaterial(materials[material.index], hitRecord, &attenuation, &emissionColor, &localRay, seed);

      if (!scattered) {
        colorStackIdx += 1;
        colorStack[colorStackIdx] = BouncingInfo(Color(0.0,0.0,0.0), emissionColor);
        break;
      } else {
        colorStackIdx += 1;
        colorStack[colorStackIdx] = BouncingInfo(attenuation, Color(0.0,0,0));
      }
    } else {
      // did not hit anything until infinity
      break;
    }
  }

  var color = Color(0,0,0);
  let lastIdx = colorStackIdx;
  for (var i = colorStackIdx; i >= 0; i -= 1) {
    let bouncing = colorStack[i];
    color = bouncing.emission + (bouncing.attenuation * color);
  }

  return color;
}

fn getRay(i: f32, j: f32, seed: ptr<function, u32>) -> Ray {
  let pixelCenter = pixel00Loc + (i * pixelDeltaU) + (j * pixelDeltaV);
  let pixelSample = pixelCenter + pixelSampleSquare(seed);

  let rayOrigin = cameraCenter;
  let rayDirection = pixelSample - cameraCenter;

  let ray = Ray(cameraCenter, rayDirection);
  return ray;
}

fn pixelSampleSquare(seed: ptr<function, u32>) -> vec3<f32> {
  let px = -0.5 + randomF32(seed);
  let py = -0.5 + randomF32(seed);
  return (px * pixelDeltaU) + (py * pixelDeltaV);
}

fn writeColor(pixelColor: vec3<f32>, x: i32, y: i32) {
  let scale = 1.0 / f32(samplesPerPixel);
  let adjustedColor = pixelColor * scale;
  textureStore(texture, vec2<i32>(x, y), vec4<f32>(adjustedColor, 1.0));
}

@compute
@workgroup_size(${maxWorkgroupDimension}, ${maxWorkgroupDimension}, 1)
fn computeMain(@builtin(global_invocation_id) local_id: vec3<u32>) {
  var seed = local_id.x + local_id.y * ${imageWidth};
  
  let i = f32(local_id.x);
  let j = f32(local_id.y);
  
  var pixelColor = vec3<f32>(0.0, 0.0, 0.0);
  
  for (var sample = 0; sample < samplesPerPixel; sample += 1) {
      let r = getRay(i, j, &seed);
      pixelColor += rayColor(r, &seed);
  }
  
  writeColor(pixelColor, i32(local_id.x), i32(local_id.y));  
}

@group(0) @binding(0) var texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> bvhNodes: array<BvhNode>;

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

const samplesPerPixel = 30;
const maxDepth = 5;

const MINIMUM_FLOAT_EPSILON = 1e-8;

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
  emissionLuminance: f32,
  emissionColor: Color,
}

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

fn renderMaterial(material: Material, hitRecord: HitRecord, attenuation: ptr<function, Color>, emissionColor: ptr<function, Color>, scattered: ptr<function, Ray>, seed: ptr<function, u32>) -> bool {
  // Inverse-square law
  let intensityFactor = material.emissionLuminance / pow(hitRecord.t, 2);
  (*emissionColor) = min(material.emissionColor * intensityFactor, Color(1.0, 1.0, 1.0));

  // todo: optimize?
  if (identical(material.baseColor, Color(0.0, 0.0, 0.0))) {
    return false;
  }
  
  var scatterDirection = hitRecord.normal + normalize(randInUnitSphere(seed));

  // Catch degenerate scatter direction
  if (nearZero(scatterDirection)) {
    scatterDirection = hitRecord.normal;
  }

  (*scattered) = Ray(hitRecord.point, scatterDirection);

  (*attenuation) = (material.baseColor * material.baseWeight);

  return true;  
}

const materials: array<Material, 4> = array<Material, 4>(
  // base materials
  Material(0.8, Color(0.5, 1.0, 0.0), 0, Color(0.0, 0.0, 0.0)),
  Material(1.0, Color(1.0, 0.5, 0.2), 0, Color(0.0, 0.0, 0.0)),
  // lights
  Material(0.0, Color(0.0, 0.0, 0.0), 10.0, Color(1.0, 0.5, 1.0)),
  Material(0.0, Color(0.0, 0.0, 0.0), 10.0, Color(1.0, 1.0, 1.0))
);

const defaultMaterial = MaterialDefinition(0);
const defaultDiffuseLightMaterial = MaterialDefinition(2);

const TRIANGLE_COUNT = 6;
// Triangles are encoded as first being the lower point, then the two edges
const triangles: array<Triangle, TRIANGLE_COUNT> = array<Triangle, TRIANGLE_COUNT>(
  // wall facing camera
  Triangle(vec3<f32>(-3, 0, -3), vec3<f32>(4, 0, 0), vec3<f32>(0, 4, 0), MaterialDefinition(1)),
  Triangle(vec3<f32>(-3, 4, -3), vec3<f32>(4, -4, 0), vec3<f32>(4, 0, 0), MaterialDefinition(1)),
  // ground floor below the wall
  Triangle(vec3<f32>(-3, 0, -3), vec3<f32>(4, 0, 0), vec3<f32>(0, -2, 1.5), defaultMaterial),
  Triangle(vec3<f32>(1, 0, -3), vec3<f32>(0, -2, 1.5), vec3<f32>(-4, -2, 1.5), defaultMaterial),
  // Light source from the right
  Triangle(vec3<f32>(2, 5, -6), vec3<f32>(0.1, -20, 0), vec3<f32>(0, 0, 10), defaultDiffuseLightMaterial),
  // Light from the left
  Triangle(vec3<f32>(-3, 5, -6), vec3<f32>(0.0, -20, 0), vec3<f32>(0, 0, 10), MaterialDefinition(3)),
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
  (*hitRecord).normal = normalize(cross(edge1, edge2));
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

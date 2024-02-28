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

type Params = {
  maxBvhStackDepth: number;
};

/**
 * expects:
 * uniformData: {
 *   objectDefinitionLength: i32,
 * }
 * @returns
 */
export function buildBvhShader({ maxBvhStackDepth }: Params) {
  return /* wgsl */ `
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
    
    let v1 = positions[v1Index];
    let v2 = positions[v2Index];
    let v3 = positions[v3Index];
    let x = v1[0];
    let y = v2[0];
    let z = v3[0];

    let normalX = v1[1];
    let normalY = v2[1];
    let normalZ = v3[1];
    
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
  `;
}

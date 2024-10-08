import { BufferGeometry, Group, Mesh, Object3D } from "three";
import * as BufferGeometryUtils from "three/addons/utils/BufferGeometryUtils.js";

function isMesh(object: Object3D): object is Mesh {
  return (object as Mesh).isMesh;
}

function traverseGroups(group: Group[]): Mesh[] {
  const meshes: Mesh[] = [];
  for (const child of group) {
    child.traverseVisible((object) => {
      if (isMesh(object)) {
        meshes.push(object);
      }
    });
  }
  return meshes;
}

export function consolidateMesh(groups: Group[]): {
  geometry: BufferGeometry;
  materials: Mesh["material"][];
} {
  const meshes = traverseGroups(groups);

  const mergedGeometry = BufferGeometryUtils.mergeGeometries(
    meshes.map((mesh) => mesh.geometry),
    true,
  );

  const materials = meshes.map((mesh) => mesh.material).flat(1);

  return {
    geometry: mergedGeometry,
    materials,
  };
}

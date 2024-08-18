import { getBVHExtremes, MeshBVH } from "three-mesh-bvh";
import { logGroup } from "./benchmark/cpu-performance-logger";
import { consolidateMesh } from "./consolidate-mesh";
import { Group } from "three";
import { assertMeshBVHInternalStructure, bvhToTextures } from "./bvh-util";
import { InternalError } from "./core/exceptions";

export function prepareGeometry(model: { scene: Group }) {
  const reducedModel = consolidateMesh([model.scene]);
  const cpuLogGroup = logGroup("cpu");
  const boundsTree = new MeshBVH(reducedModel.geometry, {
    // This property is not officially supported by three-mesh-bvh just yet
    // @ts-ignore
    indirect: true,
  });

  const isStructureMatching = assertMeshBVHInternalStructure(boundsTree);
  if (!isStructureMatching) {
    throw new InternalError(
      "MeshBVH internal structure does not match, this indicates a change in the library which is not supported at prepareGeometry.",
    );
  }
  const extremes = getBVHExtremes(boundsTree);
  const correspondingExtremesEntry = extremes[0];
  const maxBvhDepth = correspondingExtremesEntry.depth.max;

  const { boundsArray, contentsArray } = bvhToTextures(boundsTree);
  const bvhBuildTime = cpuLogGroup.end();

  const meshPositions = boundsTree.geometry.attributes.position.array;
  const positions = meshPositions;

  const meshIndices = boundsTree.geometry.index!.array;

  const normals = boundsTree.geometry.attributes.normal.array;

  return {
    indirectBuffer: boundsTree._indirectBuffer,
    boundsArray,
    contentsArray,
    positions,
    normals,
    meshIndices,
    modelGroups: reducedModel.geometry.groups,
    modelMaterials: reducedModel.materials,
    maxBvhDepth,
    bvhBuildTime,
  };
}

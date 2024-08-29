import { MeshBVH, getBVHExtremes } from "three-mesh-bvh";
import { startMeasurementGroup } from "./benchmark/performance-measurement-group.ts";
import { consolidateMesh } from "./consolidate-mesh";
import { GeometryGroup, Group } from "three";
import {
  assertMeshBVHInternalStructure,
  convertBvhToDataViews,
} from "./bvh-util";
import { InternalError, InvalidMaterialGroupError } from "./core/exceptions";

export type MaterializedGeometryGroup = {
  start: number;
  count: number;
  materialIndex: number;
};

function assertMaterializedGeometryGroup(
  geometryGroup: GeometryGroup,
): asserts geometryGroup is MaterializedGeometryGroup {
  if (
    typeof geometryGroup.start !== "number" ||
    typeof geometryGroup.count !== "number" ||
    typeof geometryGroup.materialIndex !== "number"
  ) {
    throw new InvalidMaterialGroupError(geometryGroup);
  }
}

export function prepareGeometry(model: { scene: Group }) {
  const reducedModel = consolidateMesh([model.scene]);
  const cpuLogGroup = startMeasurementGroup();
  const boundsTree = new MeshBVH(reducedModel.geometry, {
    // @ts-expect-error This property is not officially supported by three-mesh-bvh just yet
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

  const { boundsArray, contentsArray } = convertBvhToDataViews(boundsTree);
  const bvhBuildTime = cpuLogGroup.end();

  const meshPositions = boundsTree.geometry.attributes.position.array;
  const positions = meshPositions;

  const meshIndices = boundsTree.geometry.index!.array;

  const normals = boundsTree.geometry.attributes.normal.array;

  const modelGroups = reducedModel.geometry.groups.map((geometryGroup) => {
    assertMaterializedGeometryGroup(geometryGroup);
    return geometryGroup;
  });

  return {
    indirectBuffer: boundsTree._indirectBuffer,
    boundsArray,
    contentsArray,
    positions,
    normals,
    meshIndices,
    modelGroups,
    modelMaterials: reducedModel.materials,
    maxBvhDepth,
    bvhBuildTime,
  };
}

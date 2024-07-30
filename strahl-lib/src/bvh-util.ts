import { MeshBVH } from "three-mesh-bvh";

// Build constants
const BYTES_PER_NODE = 6 * 4 + 4 + 4;

// See https://github.com/gkjohnson/three-mesh-bvh/blob/0eda7b718799e1709ad9efecdcc13c06ae3d5a55/src/core/utils/nodeBufferUtils.js
function isLeaf(n16: number, uint16Array: Uint16Array) {
  return uint16Array[n16 + 15] === 0xffff;
}

function getAtOffset(n32: number, uint32Array: Uint32Array) {
  return uint32Array[n32 + 6];
}

function getCount(n16: number, uint16Array: Uint16Array) {
  return uint16Array[n16 + 14];
}

function getRightNode(n32: number, uint32Array: Uint32Array) {
  return uint32Array[n32 + 6];
}

function getSplitAxis(n32: number, uint32Array: Uint32Array) {
  return uint32Array[n32 + 7];
}

function getBoundingDataIndex(n32: number) {
  return n32;
}

type MeshBVHInternal = {
  _roots: ArrayBuffer[];
};

// CODE#BVH-TRANSFER
// Inspired by https://github.com/gkjohnson/three-mesh-bvh/blob/0eda7b718799e1709ad9efecdcc13c06ae3d5a55/src/gpu/MeshBVHUniformStruct.js#L110C1-L191C2
export function bvhToTextures(bvh: MeshBVH) {
  const privateBvh = bvh as unknown as MeshBVHInternal;
  const roots = privateBvh._roots;

  if (roots.length !== 1) {
    throw new Error("MeshBVHUniformStruct: Multi-root BVHs not supported.");
  }

  const root = roots[0];
  const uint16Array = new Uint16Array(root);
  const uint32Array = new Uint32Array(root);
  const float32Array = new Float32Array(root);

  // Both bounds need two elements per node so compute the height so it's twice as long as
  // the width so we can expand the row by two and still have a square texture
  const nodeCount = root.byteLength / BYTES_PER_NODE;
  const boundsDimension = 2 * Math.ceil(Math.sqrt(nodeCount / 2));
  const boundsArray = new Float32Array(4 * boundsDimension * boundsDimension);

  const contentsDimension = Math.ceil(Math.sqrt(nodeCount));
  const contentsArray = new Uint32Array(
    2 * contentsDimension * contentsDimension,
  );

  for (let i = 0; i < nodeCount; i++) {
    const nodeIndex32 = (i * BYTES_PER_NODE) / 4;
    const nodeIndex16 = nodeIndex32 * 2;
    const boundsIndex = getBoundingDataIndex(nodeIndex32);
    for (let b = 0; b < 3; b++) {
      boundsArray[8 * i + 0 + b] = float32Array[boundsIndex + 0 + b];
      boundsArray[8 * i + 4 + b] = float32Array[boundsIndex + 3 + b];
    }

    if (isLeaf(nodeIndex16, uint16Array)) {
      const count = getCount(nodeIndex16, uint16Array);
      const offset = getAtOffset(nodeIndex32, uint32Array);

      const mergedLeafCount = 0xffff0000 | count;
      contentsArray[i * 2 + 0] = mergedLeafCount;
      contentsArray[i * 2 + 1] = offset;
    } else {
      const rightIndex =
        (4 * getRightNode(nodeIndex32, uint32Array)) / BYTES_PER_NODE;
      const splitAxis = getSplitAxis(nodeIndex32, uint32Array);

      contentsArray[i * 2 + 0] = splitAxis;
      contentsArray[i * 2 + 1] = rightIndex;
    }
  }

  return {
    boundsArray,
    contentsArray,
    boundsDimension,
    contentsDimension,
  };
}

import { TypedArray } from "three";

export function generateIndicesBuffer(
  device: GPUDevice,
  meshIndices: TypedArray,
) {
  const indices = new Uint32Array(meshIndices);

  const indicesBuffer = device.createBuffer({
    label: "Index buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * indices.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const indicesMapped = indicesBuffer.getMappedRange();
  const indicesData = new Uint32Array(indicesMapped);
  indicesData.set(indices);
  indicesBuffer.unmap();

  return indicesBuffer;
}

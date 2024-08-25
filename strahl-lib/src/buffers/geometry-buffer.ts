import { TypedArray } from "three";

/**
 * Geometry buffer contains positions and normals interleaved
 * @param device
 * @param positions
 * @param normals
 */
export function generateGeometryBuffer(
  device: GPUDevice,
  positions: TypedArray,
  normals: TypedArray,
) {
  const sizePosition =
    Float32Array.BYTES_PER_ELEMENT * (positions.length / 3) * 4;
  const sizeNormal = Float32Array.BYTES_PER_ELEMENT * (normals.length / 3) * 4;

  const geometryBuffer = device.createBuffer({
    label: "Geometry buffer",
    size: sizePosition + sizeNormal,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const geometryMapped = geometryBuffer.getMappedRange();
  const geometryData = new Float32Array(geometryMapped);

  const itemsPerVertex = 3;

  for (let i = 0; i < positions.length; i += itemsPerVertex) {
    const offset = (i / itemsPerVertex) * 4 * 2;
    const offsetNormal = offset + 4;
    geometryData.set(positions.slice(i, i + itemsPerVertex), offset);
    geometryData.set(normals.slice(i, i + itemsPerVertex), offsetNormal);
  }

  geometryBuffer.unmap();

  return geometryBuffer;
}

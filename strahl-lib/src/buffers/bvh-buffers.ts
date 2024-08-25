export function generateBvhBuffers(
  device: GPUDevice,
  boundsArray: Float32Array,
  contentsArray: Uint32Array,
  indirectBufferData: ArrayLike<number>,
) {
  // Prepare BVH Bounds
  const boundsBuffer = device.createBuffer({
    label: "BVH bounds buffer",
    size: Float32Array.BYTES_PER_ELEMENT * boundsArray.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const boundsMapped = boundsBuffer.getMappedRange();
  const boundsData = new Float32Array(boundsMapped);
  boundsData.set(boundsArray);
  boundsBuffer.unmap();

  // Prepare BVH Contents
  const contentsBuffer = device.createBuffer({
    label: "BVH contents buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * contentsArray.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const contentsMapped = contentsBuffer.getMappedRange();
  const contentsData = new Uint32Array(contentsMapped);
  contentsData.set(contentsArray);
  contentsBuffer.unmap();

  // Prepare BVH indirect buffer
  const indirectBuffer = device.createBuffer({
    label: "BVH indirect buffer",
    size: Uint32Array.BYTES_PER_ELEMENT * indirectBufferData.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const indirectMapped = indirectBuffer.getMappedRange();
  const indirectData = new Uint32Array(indirectMapped);
  indirectData.set(indirectBufferData);
  indirectBuffer.unmap();

  return {
    boundsBuffer,
    contentsBuffer,
    indirectBuffer,
  };
}

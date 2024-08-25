export function generateObjectDefinitionBuffer(
  device: GPUDevice,
  modelGroups: Array<{ start: number; count: number; materialIndex: number }>,
) {
  const OBJECT_DEFINITION_SIZE_PER_ENTRY = Uint32Array.BYTES_PER_ELEMENT * 3;

  const objectDefinitionsBuffer = device.createBuffer({
    label: "Object definitions buffer",
    size: OBJECT_DEFINITION_SIZE_PER_ENTRY * modelGroups.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  const objectDefinitionsMapped = objectDefinitionsBuffer.getMappedRange();
  const objectDefinitionsData = new Uint32Array(objectDefinitionsMapped);

  objectDefinitionsData.set(
    modelGroups.map((g) => [g.start, g.count, g.materialIndex]).flat(1),
  );
  objectDefinitionsBuffer.unmap();

  return objectDefinitionsBuffer;
}

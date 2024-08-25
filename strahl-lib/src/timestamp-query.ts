export type TimestampQueryContext = {
  querySet: GPUQuerySet;
  resolveBuffer: GPUBuffer;
  resultBuffer: GPUBuffer;
};

export function generateTimestampQuery(
  device: GPUDevice,
): TimestampQueryContext {
  const timestampQuerySet = device.createQuerySet({
    type: "timestamp",
    count: 2,
  });

  const timestampQueryResolveBuffer = device.createBuffer({
    size: timestampQuerySet.count * 8,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.QUERY_RESOLVE,
  });

  const timestampQueryResultBuffer = device.createBuffer({
    size: timestampQueryResolveBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  return {
    querySet: timestampQuerySet,
    resolveBuffer: timestampQueryResolveBuffer,
    resultBuffer: timestampQueryResultBuffer,
  };
}

export function encodeTimestampQuery(
  encoder: GPUCommandEncoder,
  context: TimestampQueryContext,
) {
  encoder.resolveQuerySet(context.querySet, 0, 2, context.resolveBuffer, 0);

  if (context.resultBuffer.mapState === "unmapped") {
    encoder.copyBufferToBuffer(
      context.resolveBuffer,
      0,
      context.resultBuffer,
      0,
      context.resolveBuffer.size,
    );
  }
}

export async function retrieveTimestampQueryTime(
  context: TimestampQueryContext,
) {
  if (context.resultBuffer.mapState !== "unmapped") {
    return null;
  }
  await context.resultBuffer.mapAsync(GPUMapMode.READ);

  const data = new BigUint64Array(context.resultBuffer.getMappedRange());
  const gpuTime = data[1] - data[0];
  context.resultBuffer.unmap();
  return gpuTime;
}

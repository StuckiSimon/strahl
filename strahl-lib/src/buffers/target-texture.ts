import { OutputSizeConfiguration } from "../path-tracer.ts";

export function prepareTargetTexture(
  device: GPUDevice,
  { width, height }: OutputSizeConfiguration,
) {
  return device.createTexture({
    size: [width, height],
    format: "rgba32float",
    usage:
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.COPY_SRC |
      GPUTextureUsage.STORAGE_BINDING, // Permit writing to texture in compute shader
  });
}

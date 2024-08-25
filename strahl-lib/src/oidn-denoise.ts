import { initUNetFromURL } from "oidn-web";

export async function oidnDenoise(
  {
    device,
    adapterInfo,
    url,
  }: { device: GPUDevice; adapterInfo: GPUAdapterInfo; url: string },
  {
    colorBuffer,
    albedoBuffer,
    normalBuffer,
  }: {
    colorBuffer: GPUBuffer;
    albedoBuffer: GPUBuffer;
    normalBuffer: GPUBuffer;
  },
  size: { width: number; height: number },
) {
  const unet = await initUNetFromURL(
    url,
    {
      device,
      adapterInfo,
    },
    {
      aux: true,
      hdr: true,
    },
  );

  type GPUImageData = {
    data: GPUBuffer;
    width: number;
    height: number;
  };

  return new Promise<GPUImageData>((resolve) => {
    unet.tileExecute({
      color: {
        data: colorBuffer,
        width: size.width,
        height: size.height,
      },
      albedo: {
        data: albedoBuffer,
        width: size.width,
        height: size.height,
      },
      normal: {
        data: normalBuffer,
        width: size.width,
        height: size.height,
      },
      done(finalBuffer) {
        resolve(finalBuffer);
      },
    });
  });
}

import { initUNetFromURL } from "oidn-web";
import { buildDenoisePassShader } from "./shaders/denoise-pass-shader.ts";
import { makeShaderDataDefinitions, makeStructuredView } from "webgpu-utils";
import { OIDNConfig } from "./path-tracer.ts";

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

export async function denoisePass(
  device: GPUDevice,
  adapterInfo: GPUAdapterInfo,
  oidnConfig: OIDNConfig,
  maxBvhDepth: number,
  maxWorkgroupDimension: number,
  width: number,
  height: number,
  uniformDataContent: Record<string, number | string | number[]>,
  computeBindGroupLayout: GPUBindGroupLayout,
  computeBindGroup: GPUBindGroup,
  executeRenderPass: (texture: GPUTexture, encoder: GPUCommandEncoder) => void,
  isHalted: () => boolean,
  writeTexture: GPUTexture,
  readTexture: GPUTexture,
) {
  const dynamicComputeBindGroupLayout = device.createBindGroupLayout({
    label: "Dynamic denoise pass compute bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: {
          format: "rgba32float",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { format: "rgba32float", access: "read-only" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "uniform",
        },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const denoisePassShaderCode = buildDenoisePassShader({
    bvhParams: {
      maxBvhStackDepth: maxBvhDepth,
    },
  });

  const denoisePassDefinitions = makeShaderDataDefinitions(
    denoisePassShaderCode,
  );
  const { size: bytesForUniform } = denoisePassDefinitions.uniforms.uniformData;
  const uniformData = makeStructuredView(
    denoisePassDefinitions.uniforms.uniformData,
    new ArrayBuffer(bytesForUniform),
  );

  const buildDenoisePassUniformBuffer = (mode: 0 | 1) => {
    const uniformBuffer = device.createBuffer({
      label: "Denoise pass uniform data buffer",
      size: bytesForUniform,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    uniformData.set({
      ...uniformDataContent,
      mode,
    });
    device.queue.writeBuffer(uniformBuffer, 0, uniformData.arrayBuffer);

    return uniformBuffer;
  };

  const normalUniformBuffer = buildDenoisePassUniformBuffer(0);
  const albedoUniformBuffer = buildDenoisePassUniformBuffer(1);

  const computePipelineLayout = device.createPipelineLayout({
    label: "Dynamic denoise pass compute pipeline layout",
    bindGroupLayouts: [computeBindGroupLayout, dynamicComputeBindGroupLayout],
  });

  const float32ArrayImageSize = width * height * 4;

  const normalImageBuffer = device.createBuffer({
    label: "Normal image buffer",
    size: Float32Array.BYTES_PER_ELEMENT * float32ArrayImageSize,
    usage: GPUBufferUsage.STORAGE,
  });

  const albedoImageBuffer = device.createBuffer({
    label: "Albedo image buffer",
    size: Float32Array.BYTES_PER_ELEMENT * float32ArrayImageSize,
    usage: GPUBufferUsage.STORAGE,
  });

  const computeShaderModule = device.createShaderModule({
    label: "Denoise Pass Compute Shader",
    code: denoisePassShaderCode,
  });

  const computePipeline = device.createComputePipeline({
    label: "Denoise Pass Compute pipeline",
    layout: computePipelineLayout,
    compute: {
      module: computeShaderModule,
      entryPoint: "computeMain",
      constants: {
        wgSize: maxWorkgroupDimension,
        imageWidth: width,
        imageHeight: height,
      },
    },
  });

  const executeDenoisePass = (
    imageBuffer: GPUBuffer,
    uniformBuffer: GPUBuffer,
  ) => {
    const dynamicComputeBindGroup = device.createBindGroup({
      label: "Dynamic denoise pass compute bind group",
      layout: dynamicComputeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: writeTexture.createView(),
        },
        {
          binding: 1,
          resource: readTexture.createView(),
        },
        {
          binding: 2,
          resource: {
            buffer: uniformBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: imageBuffer,
          },
        },
      ],
    });
    const encoder = device.createCommandEncoder();

    const computePass = encoder.beginComputePass();
    computePass.setBindGroup(0, computeBindGroup);
    computePass.setBindGroup(1, dynamicComputeBindGroup);

    computePass.setPipeline(computePipeline);

    const dispatchX = Math.ceil(width / maxWorkgroupDimension);
    const dispatchY = Math.ceil(height / maxWorkgroupDimension);
    computePass.dispatchWorkgroups(dispatchX, dispatchY);

    computePass.end();

    device.queue.submit([encoder.finish()]);
  };

  executeDenoisePass(normalImageBuffer, normalUniformBuffer);
  executeDenoisePass(albedoImageBuffer, albedoUniformBuffer);

  const textureBuffer = device.createBuffer({
    label: "Texture buffer",
    usage:
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.STORAGE,
    size: Float32Array.BYTES_PER_ELEMENT * 4 * width * height,
  });

  {
    const encoder = device.createCommandEncoder();

    encoder.copyTextureToBuffer(
      { texture: writeTexture },
      { buffer: textureBuffer, bytesPerRow: width * 4 * 4 },
      [width, height],
    );

    device.queue.submit([encoder.finish()]);
  }

  if (isHalted()) {
    return;
  }
  const outputBuffer = await oidnDenoise(
    { device, adapterInfo: adapterInfo, url: oidnConfig.url },
    {
      colorBuffer: textureBuffer,
      albedoBuffer: albedoImageBuffer,
      normalBuffer: normalImageBuffer,
    },
    {
      width,
      height,
    },
  );

  if (isHalted()) {
    return;
  }

  {
    const textureFinal = device.createTexture({
      size: [width, height],
      format: "rgba32float",
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.COPY_SRC,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToTexture(
      { buffer: outputBuffer.data, bytesPerRow: width * 4 * 4 },
      { texture: textureFinal },
      [width, height],
    );
    device.queue.submit([encoder.finish()]);

    const encoder2 = device.createCommandEncoder();
    executeRenderPass(textureFinal, encoder2);
  }

  textureBuffer.unmap();
}

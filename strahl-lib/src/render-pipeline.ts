import { GaussianConfig, OutputSizeConfiguration } from "./path-tracer.ts";
import { buildRenderShader } from "./shaders/render-shader.ts";
import { makeShaderDataDefinitions, makeStructuredView } from "webgpu-utils";

export function setupRenderPipeline(
  device: GPUDevice,
  useFloatTextureFiltering: boolean,
  gaussianConfig: GaussianConfig,
  { width, height }: OutputSizeConfiguration,
  format: GPUTextureFormat,
) {
  const renderShaderCode = buildRenderShader();

  const renderShaderModule = device.createShaderModule({
    label: "Render Shader",
    code: renderShaderCode,
  });

  const renderShaderDefinitions = makeShaderDataDefinitions(renderShaderCode);
  const { size: bytesForRenderUniform } =
    renderShaderDefinitions.uniforms.uniformData;
  const renderUniformData = makeStructuredView(
    renderShaderDefinitions.uniforms.uniformData,
    new ArrayBuffer(bytesForRenderUniform),
  );

  const renderUniformBuffer = device.createBuffer({
    label: "Render uniform data buffer",
    size: bytesForRenderUniform,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  const setRenderUniformData = (enableDenoise: boolean) => {
    renderUniformData.set({
      textureWidth: width,
      textureHeight: height,
      denoiseSigma: gaussianConfig.sigma,
      denoiseKSigma: gaussianConfig.kSigma,
      denoiseThreshold: gaussianConfig.threshold,
      enableDenoise: enableDenoise ? 1 : 0,
    });
    device.queue.writeBuffer(
      renderUniformBuffer,
      0,
      renderUniformData.arrayBuffer,
    );
  };
  setRenderUniformData(false);

  const renderBindGroupLayout = device.createBindGroupLayout({
    label: "Texture sampler bind group layout",
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: {
          type: useFloatTextureFiltering ? "filtering" : "non-filtering",
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        texture: {
          sampleType: useFloatTextureFiltering ? "float" : "unfilterable-float",
        },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: {
          type: "uniform",
        },
      },
    ],
  });

  const renderPipelineLayout = device.createPipelineLayout({
    label: "Pipeline Layout",
    bindGroupLayouts: [renderBindGroupLayout],
  });

  const renderPipeline = device.createRenderPipeline({
    label: "Render pipeline",
    layout: renderPipelineLayout,
    vertex: {
      module: renderShaderModule,
      entryPoint: "vertexMain",
      buffers: [],
    },
    fragment: {
      module: renderShaderModule,
      entryPoint: "fragmentMain",
      targets: [
        {
          format,
        },
      ],
    },
  });

  const sampler = device.createSampler({
    magFilter: useFloatTextureFiltering ? "linear" : "nearest",
  });

  return {
    setRenderUniformData,
    executeRenderPass: (
      context: GPUCanvasContext,
      texture: GPUTexture,
      encoder: GPUCommandEncoder,
    ) => {
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          {
            view: context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0, g: 0, b: 0.2, a: 1 },
            storeOp: "store",
          },
        ],
      });

      pass.setPipeline(renderPipeline);

      const renderBindGroup = device.createBindGroup({
        label: "Texture sampler bind group",
        layout: renderBindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: sampler,
          },
          {
            binding: 1,
            resource: texture.createView(),
          },
          {
            binding: 2,
            resource: {
              buffer: renderUniformBuffer,
            },
          },
        ],
      });

      pass.setBindGroup(0, renderBindGroup);
      const RENDER_TEXTURE_VERTEX_COUNT = 6;
      pass.draw(RENDER_TEXTURE_VERTEX_COUNT);

      pass.end();

      const commandBuffer = encoder.finish();

      device.queue.submit([commandBuffer]);
    },
  };
}

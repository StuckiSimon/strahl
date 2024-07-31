import { logGroup } from "./cpu-performance-logger";

const MODEL_URL_FULL = "models/series-61-rotated/61-serie-edit.gltf"; // 1'068'735
const MODEL_URL_BARE_BONES = "models/series-61-simplified-2/61-serie-edit.gltf"; // 10'687
const MODEL_URL_MID = "models/series-61-simplified/61-serie-edit.gltf"; // 106'873

async function run(target: number, yielder: any) {
  const renderTimeElement = document.createElement("div");
  renderTimeElement.textContent = `Finished setup`;
  document.body.appendChild(renderTimeElement);

  const TARGET_SAMPLES = 0; //199; // 200

  console.info("start rendering", window.performance.now());
  const buildRenderLoop = () => {
    const renderLoopStart = logGroup("render loop full");
    let state: "running" | "halted" = "running";

    const isHalted = () => state === "halted";

    let currentAnimationFrameRequest: number | null = null;
    let currentSample = 0;
    let renderAgg = 0;
    let renderTimes = [];

    const render = async () => {};
    currentAnimationFrameRequest = requestAnimationFrame(render);

    return {
      terminateLoop: () => {
        state = "halted";
        if (currentAnimationFrameRequest !== null) {
          cancelAnimationFrame(currentAnimationFrameRequest);
        }
      },
    };
  };

  buildRenderLoop();

  /*controls.addEventListener("change", () => {
    renderLoop.terminateLoop();

    renderLoop = buildRenderLoop();
  });*/
}

function getStandardDeviation(list: number[]) {
  if (list.length === 0) {
    return 0;
  }
  const n = list.length;
  const mean = list.reduce((a, b) => a + b) / n;
  return Math.sqrt(
    list.map((x) => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / n,
  );
}

function getSampleMean(list: number[]) {
  if (list.length === 0) {
    return 0;
  }
  return list.reduce((a, b) => a + b) / list.length;
}

function confidenceInterval(xBar: number, s: number, n: number, z: number) {
  const marginOfError = z * (s / Math.sqrt(n));

  const lowerBound = xBar - marginOfError;
  const upperBound = xBar + marginOfError;

  return { lowerBound, upperBound, marginOfError };
}

const userAgent = window.navigator.userAgent;

// @ts-ignore
function getLimits(context) {
  return {
    maxBindGroups: context.maxBindGroups,
    maxBindGroupsPlusVertexBuffers: context.maxBindGroupsPlusVertexBuffers,
    maxBindingsPerBindGroup: context.maxBindingsPerBindGroup,
    maxBufferSize: context.maxBufferSize,
    maxColorAttachmentBytesPerSample: context.maxColorAttachmentBytesPerSample,
    maxColorAttachments: context.maxColorAttachments,
    maxComputeInvocationsPerWorkgroup:
      context.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: context.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: context.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: context.maxComputeWorkgroupSizeZ,
    maxComputeWorkgroupStorageSize: context.maxComputeWorkgroupStorageSize,
    maxComputeWorkgroupsPerDimension: context.maxComputeWorkgroupsPerDimension,
    maxDynamicStorageBuffersPerPipelineLayout:
      context.maxDynamicStorageBuffersPerPipelineLayout,
    maxDynamicUniformBuffersPerPipelineLayout:
      context.maxDynamicUniformBuffersPerPipelineLayout,
    maxInterStageShaderComponents: context.maxInterStageShaderComponents,
    maxInterStageShaderVariables: context.maxInterStageShaderVariables,
    maxSampledTexturesPerShaderStage: context.maxSampledTexturesPerShaderStage,
    maxSamplersPerShaderStage: context.maxSamplersPerShaderStage,
    maxStorageBufferBindingSize: context.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: context.maxStorageBuffersPerShaderStage,
    maxStorageTexturesPerShaderStage: context.maxStorageTexturesPerShaderStage,
    maxTextureArrayLayers: context.maxTextureArrayLayers,
    maxTextureDimension1D: context.maxTextureDimension1D,
    maxTextureDimension2D: context.maxTextureDimension2D,
    maxTextureDimension3D: context.maxTextureDimension3D,
    maxUniformBufferBindingSize: context.maxUniformBufferBindingSize,
    maxUniformBuffersPerShaderStage: context.maxUniformBuffersPerShaderStage,
    maxVertexAttributes: context.maxVertexAttributes,
    maxVertexBufferArrayStride: context.maxVertexBufferArrayStride,
    maxVertexBuffers: context.maxVertexBuffers,
    minStorageBufferOffsetAlignment: context.minStorageBufferOffsetAlignment,
    minUniformBufferOffsetAlignment: context.minUniformBufferOffsetAlignment,
  };
}

async function initWebGPUReport() {
  if (!navigator.gpu) {
    return {
      userAgent,
      "no-webgpu": true,
    };
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return {
      userAgent,
      "no-adapter": true,
    };
  }

  const device = await adapter.requestDevice();

  const info = await adapter.requestAdapterInfo();

  console.log("adapter", adapter);
  console.log("device", device);

  return {
    userAgent,
    adapterInfo: {
      isFallbackAdapter: adapter.isFallbackAdapter,
      architecture: info.architecture,
      description: info.description,
      device: info.device,
      vendor: info.vendor,
      features: Array.from(adapter.features.keys()),
      limits: getLimits(adapter.limits),
    },
    deviceInfo: {
      label: device.label,
      features: Array.from(device.features.keys()),
      limits: getLimits(device.limits),
    },
  };
}

function saveTextAsFile(text: string, filename: string) {
  const blob = new Blob([text], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = filename;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function main() {
  document.getElementById("reset-button")?.addEventListener("click", () => {
    localStorage.removeItem("strahl-path-tracer-state");
    window.location.reload();
  });

  document.getElementById("start-button")?.addEventListener("click", () => {
    localStorage.setItem(
      "strahl-path-tracer-state",
      JSON.stringify({
        max: [],
        mid: [],
        min: [],
      }),
    );
    window.location.reload();
  });

  // @ts-ignore
  let state = JSON.parse(localStorage.getItem("strahl-path-tracer-state"));
  console.log(state);

  if (state === null) {
    console.log("not yet started");
    return;
  }

  const RUNS = 30;
  const totalRuns = RUNS * 3;
  const hasStillSomeRunsToDo =
    state.max.length < RUNS ||
    state.mid.length < RUNS ||
    state.min.length < RUNS;

  const runsAlreadyDone =
    state.max.length + state.mid.length + state.min.length;

  const reportDiv = document.createElement("div");
  let text = `${runsAlreadyDone} / ${totalRuns} runs already done`;
  if (!hasStillSomeRunsToDo) {
    text = `All runs done, benchmark finished ✅`;
  }
  reportDiv.textContent = text;
  document.body.appendChild(reportDiv);

  if (!hasStillSomeRunsToDo) {
    const gpuInfo = await initWebGPUReport();

    const fullReport = { ...gpuInfo, ...state };

    const downloadButton = document.createElement("button");
    downloadButton.textContent = "Download report";
    downloadButton.addEventListener("click", () => {
      saveTextAsFile(
        JSON.stringify(fullReport, null, 2),
        "benchmark-report.json",
      );
    });
    document.body.appendChild(downloadButton);

    const pre = document.createElement("pre");
    pre.style.height = "calc(100vh - 200px)";
    pre.style.overflow = "auto";
    pre.textContent = JSON.stringify(fullReport, null, 2);
    document.body.appendChild(pre);

    return;
  }

  const itemWithFewestRuns = Math.min(
    state.max.length,
    state.mid.length,
    state.min.length,
  );
  const target =
    itemWithFewestRuns === state.max.length
      ? 0
      : itemWithFewestRuns === state.mid.length
        ? 1
        : 2;

  run(target, (report: any) => {
    console.log(report);
    const deviation = getStandardDeviation(report.renderTimes);
    const mean = getSampleMean(report.renderTimes);
    console.log("Standard deviation", deviation);
    console.log("Mean", mean);
    const { lowerBound, upperBound, marginOfError } = confidenceInterval(
      mean,
      deviation,
      report.renderTimes.length,
      1.96, // 95% confidence interval
    );
    console.log(
      `With 95% confidence, the true mean is between ${lowerBound} and ${upperBound} (+- ${marginOfError})`,
    );

    const reportPot = target === 0 ? "max" : target === 1 ? "mid" : "min";

    state[reportPot].push({
      ...report,
      stats: {
        deviation,
        mean,
        lowerBound,
        upperBound,
        marginOfError,
      },
    });
    localStorage.setItem("strahl-path-tracer-state", JSON.stringify(state));

    window.location.reload();
  });
}

main();

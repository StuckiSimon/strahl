import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { logGroup } from "./cpu-performance-logger.ts";
import { OpenPBRMaterial } from "../openpbr-material.ts";
import runPathTracer, { PathTracerOptions } from "../path-tracer.ts";
import getStatsForReportStructure from "./benchmark-analyser.ts";
import { isNil } from "../util/is-nil.ts";
import {
  getConfidenceInterval,
  getSampleMean,
  getStandardDeviation,
} from "./maths.ts";

const MODEL_URL_FULL = "models/series-61-rotated/61-serie-edit.gltf"; // 1'068'735
const MODEL_URL_BARE_BONES = "models/series-61-simplified-2/61-serie-edit.gltf"; // 10'687
const MODEL_URL_MID = "models/series-61-simplified/61-serie-edit.gltf"; // 106'873

const gltfLoader = new GLTFLoader();

async function loadGltf(url: string) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

const defaultBlueMaterial = new OpenPBRMaterial();
defaultBlueMaterial.oBaseColor = [0.0, 0.9, 1.0];

const plasticYellow = new OpenPBRMaterial();
plasticYellow.oBaseColor = [0.9, 0.7, 0.23];
plasticYellow.oSpecularWeight = 0.01;

const greenPlastic = new OpenPBRMaterial();
greenPlastic.oBaseColor = [0.4, 0.6, 0.33];

const redPlasticMaterial = new OpenPBRMaterial();
redPlasticMaterial.oBaseColor = [0.67, 0.18, 0.12];
redPlasticMaterial.oSpecularColor = [1.0, 1.0, 0.9];
redPlasticMaterial.oSpecularWeight = 0.2;

const metalTestMaterial = new OpenPBRMaterial();
metalTestMaterial.oBaseColor = [0.9, 0.9, 0.9];
metalTestMaterial.oBaseMetalness = 1.0;

const metalMaterial = new OpenPBRMaterial();
metalMaterial.oBaseColor = [0.9, 0.9, 0.9];
metalMaterial.oBaseMetalness = 1.0;
metalMaterial.oBaseDiffuseRoughness = 0.5;

const roughMetalMaterial = new OpenPBRMaterial();
roughMetalMaterial.oBaseColor = [0.9, 0.9, 0.9];

const roughMetalMaterial2 = new OpenPBRMaterial();
roughMetalMaterial2.oBaseColor = [0.9, 0.9, 0.9];
roughMetalMaterial2.oSpecularWeight = 0.0;
roughMetalMaterial.oBaseMetalness = 1.0;

const roughMetalMaterial3 = new OpenPBRMaterial();
roughMetalMaterial3.oBaseColor = [0.4, 0.4, 0.4];
roughMetalMaterial3.oBaseMetalness = 0.4;
roughMetalMaterial3.oSpecularWeight = 0.2;

const copperMetalMaterial = new OpenPBRMaterial();
copperMetalMaterial.oBaseColor = [0.9, 0.6, 0.4];
copperMetalMaterial.oBaseMetalness = 1.0;

const blackPlastic = new OpenPBRMaterial();
blackPlastic.oBaseColor = [0.25, 0.25, 0.25];

const whitePlastic = new OpenPBRMaterial();
whitePlastic.oBaseColor = [0.8, 0.8, 0.8];

const greyPlastic = new OpenPBRMaterial();
greyPlastic.oBaseColor = [0.5, 0.5, 0.5];
greyPlastic.oSpecularWeight = 0.0;

const beigeClayMaterial = new OpenPBRMaterial();
beigeClayMaterial.oBaseColor = [0.9, 0.8, 0.7];
beigeClayMaterial.oBaseMetalness = 0.0;
beigeClayMaterial.oBaseDiffuseRoughness = 0.0;

const MATERIAL_MAP = {
  material_name_kunststoff_gelbFCBE37_rau_2_mtl: plasticYellow,
  material_name_kunststoff_gelbFCBE37_rau_3_mtl: plasticYellow,
  material_name_kunststoff_verkehrsrotB81D12_rau_5_mtl: redPlasticMaterial,
  material_name_kunststoff_gruen9CC289_glanz_SSS_0_mtl: greenPlastic,
  material_name_kunststoff_schwarz141414_rau_1_mtl: blackPlastic,
  material_name_metall_stahl67686A_abgenutzt_7_mtl: metalMaterial,
  material_name_kunststoff_weissDCDCDC_rau_6_mtl: whitePlastic,
  // serie 45
  material_name_kunststoff_schwarz000000_rau_0_mtl: blackPlastic,
  material_name_kunststoff_schwarz000000_rau_31_mtl: blackPlastic,
  material_name_metall_stahl67686A_abgenutzt_32_mtl: roughMetalMaterial,
  material_name_kunststoff_verkehrsrotB81D12_rau_33_mtl: redPlasticMaterial,
  material_name_kunststoff_weissDCDCDC_transparent_trueb_34_mtl: whitePlastic,
  // serie 45 cleaned
  material_name_kunststoff_verkehrsrotB81D12_rau_2_mtl: redPlasticMaterial,
  material_name_kunststoff_verkehrsrotB81D12_rau_3_mtl: redPlasticMaterial,
  material_name_kunststoff_weissDCDCDC_transparent_trueb_4_mtl: whitePlastic,
  material_name_kunststoff_gelbFCBE37_rau_5_mtl: plasticYellow,
  material_name_kunststoff_schwarz000000_rau_6_mtl: blackPlastic,
  material_name_kunststoff_gruen006054_rau_7_mtl: greenPlastic,
  material_name_metall_stahl737364_abgenutzt_rau_8_mtl: metalMaterial,
  material_name_metall_stahl909086_abgenutzt_rauer_9_mtl: metalMaterial,
  material_name_kunststoff_weissDCDCDC_transparent_trueb_10_mtl: whitePlastic,
  material_name_kunststoff_schwarz000000_rau_11_mtl: blackPlastic,
  material_name_metall_stahl67686A_abgenutzt_12_mtl: roughMetalMaterial,
  material_name_kunststoff_verkehrsrotB81D12_rau_13_mtl: redPlasticMaterial,
  material_name_kunststoff_weissDCDCDC_transparent_trueb_14_mtl: whitePlastic,
  material_name_kunststoff_schwarz000000_rau_15_mtl: blackPlastic,
  material_name_kunststoff_schwarz000000_rau_16_mtl: blackPlastic,
  material_name_metall_stahl67686A_abgenutzt_17_mtl: roughMetalMaterial,
  material_name_kunststoff_weissDCDCDC_transparent_trueb_18_mtl: whitePlastic,
};

type FinishedSamplingOptions = Parameters<
  Exclude<PathTracerOptions["finishedSampling"], undefined>
>[0] & {
  fullRunTime: number;
};

async function run(
  target: number,
  yielder: (params: FinishedSamplingOptions) => void | Promise<void>,
) {
  const runStartGroup = logGroup("full-model");

  const MODEL_URL =
    target === 0
      ? MODEL_URL_FULL
      : target === 1
        ? MODEL_URL_MID
        : MODEL_URL_BARE_BONES;

  const model: any = await loadGltf(MODEL_URL);

  model.scene.traverseVisible((object: any) => {
    if (object.material === undefined) {
      return;
    }
    const materialName = object.material.name;
    if (materialName in MATERIAL_MAP) {
      object.material = MATERIAL_MAP[materialName as keyof typeof MATERIAL_MAP];
    } else {
      console.log(materialName);
      object.material = defaultBlueMaterial;
    }
  });

  runPathTracer("render-target", model, {
    targetSamples: 99,
    size: 512,
    viewProjectionConfiguration: {
      matrixWorldContent: [
        -0.32948748533091665, -2.7755575615628914e-17, 0.9441599424940185, 0,
        -0.39564488300979206, 0.9079657694010268, -0.1380698668941539, 0,
        -0.8572649086242106, -0.4190443432335072, -0.29916335812649514, 0,
        -167.11821635612353, -83.14592338588088, -43.26162715863958, 1,
      ],
      cameraTargetDistance: 200,
      fov: 23.6701655,
      aspect: 1,
    },
    finishedSampling: (params) => {
      const fullRunTime = runStartGroup.end();
      yielder({
        fullRunTime,
        ...params,
      });
    },
  });
}

const userAgent = window.navigator.userAgent;

function getLimits(context: GPUSupportedLimits) {
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

const STORAGE_KEY = "strahl-path-tracer-state";

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
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  });

  document.getElementById("start-button")?.addEventListener("click", () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        max: [],
        mid: [],
        min: [],
      }),
    );
    window.location.reload();
  });

  document
    .getElementById("start-button-high-only")
    ?.addEventListener("click", () => {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          max: [],
        }),
      );
      window.location.reload();
    });

  document
    .getElementById("file-input")
    ?.addEventListener("change", async (event) => {
      const file = (event.target as HTMLInputElement).files?.[0];
      if (!file) {
        return;
      }
      const text = await file.text();
      const report = JSON.parse(text);
      console.log(report);

      localStorage.setItem(STORAGE_KEY, text);
      window.location.reload();
    });

  const stateSerialized = localStorage.getItem(STORAGE_KEY);

  if (isNil(stateSerialized)) {
    console.log("not yet started");
    return;
  }

  let state = JSON.parse(stateSerialized);
  console.log(state);

  if (state === null) {
    console.log("not yet started");
    return;
  }

  const runKeys = ["max", "mid", "min"];
  const activeRuns = runKeys.filter((key) => Array.isArray(state[key]));

  const RUNS = 30;
  const totalRuns = RUNS * activeRuns.length;
  const hasStillSomeRunsToDo = activeRuns.some(
    (key) => state[key].length < RUNS,
  );

  const runsAlreadyDone = activeRuns.reduce(
    (acc, key) => acc + state[key].length,
    0,
  );

  const reportDiv = document.createElement("div");
  let text = `${runsAlreadyDone} / ${totalRuns} runs already done`;
  if (!hasStillSomeRunsToDo) {
    text = `All runs done, benchmark finished ✅`;
  }
  reportDiv.textContent = text;
  document.getElementById("result-container")?.appendChild(reportDiv);

  if (!hasStillSomeRunsToDo) {
    const gpuInfo = await initWebGPUReport();

    const fullReport = { ...gpuInfo, ...state };

    const stats = getStatsForReportStructure(fullReport);
    console.log(stats);
    console.table(stats);

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

  const canvas = document.createElement("canvas");
  canvas.id = "render-target";
  canvas.width = 512;
  canvas.height = 512;
  document.getElementById("canvas-container")?.appendChild(canvas);

  const itemWithFewestRuns = Math.min(
    state.max?.length ?? Infinity,
    state.mid?.length ?? Infinity,
    state.min?.length ?? Infinity,
  );

  const target =
    itemWithFewestRuns === state.max?.length
      ? 0
      : itemWithFewestRuns === state.mid?.length
        ? 1
        : 2;

  run(target, (report) => {
    console.log(report);
    const deviation = getStandardDeviation(report.renderTimes);
    const mean = getSampleMean(report.renderTimes);
    console.log("Standard deviation", deviation);
    console.log("Mean", mean);
    const { lowerBound, upperBound, marginOfError } = getConfidenceInterval(
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
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));

    window.location.reload();
  });
}

main();

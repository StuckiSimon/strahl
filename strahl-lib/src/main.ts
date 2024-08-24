import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OpenPBRMaterial } from "./openpbr-material";
import runPathTracer from "./path-tracer";
import { EnvironmentLightConfig } from "./environment-light";

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
plasticYellow.oSpecularRoughness = 0.2;

const greenPlastic = new OpenPBRMaterial();
greenPlastic.oBaseColor = [0.2, 0.35, 0.2];

const redPlasticMaterial = new OpenPBRMaterial();
redPlasticMaterial.oBaseColor = [0.67, 0.18, 0.12];
redPlasticMaterial.oSpecularColor = [1.0, 1.0, 0.9];
redPlasticMaterial.oSpecularWeight = 1.0;
redPlasticMaterial.oBaseDiffuseRoughness = 0.0;

const metalTestMaterial = new OpenPBRMaterial();
metalTestMaterial.oBaseColor = [0.9, 0.9, 0.9];
metalTestMaterial.oBaseMetalness = 1.0;

const metalMaterial = new OpenPBRMaterial();
metalMaterial.oBaseColor = [0.9, 0.9, 0.9];
metalMaterial.oBaseMetalness = 1.0;
metalMaterial.oBaseDiffuseRoughness = 0.5;

const roughMetalMaterial = new OpenPBRMaterial();
roughMetalMaterial.oBaseColor = [0.9, 0.9, 0.9];
roughMetalMaterial.oBaseMetalness = 0.1;

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
blackPlastic.oSpecularWeight = 1.0;

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
  material_name_metall_stahl67686A_abgenutzt_32_mtl: roughMetalMaterial2,
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
  // metal-wall model
  metal_wall_material: (() => {
    const metalMaterial = new OpenPBRMaterial();
    metalMaterial.oBaseColor = [0.9, 0.9, 0.9];
    metalMaterial.oBaseMetalness = 1.0;
    metalMaterial.oSpecularRoughness = 0.1;
    metalMaterial.oBaseDiffuseRoughness = 0.0;
    return metalMaterial;
  })(),
  material_name_metall_stahl737364_abgenutzt_rau_1_mtl: roughMetalMaterial3,
  material_name_kunststoff_weissDCDCDC_transparent_trueb_3_mtl: greyPlastic,
  material_name_kunststoff_gelbFCBE37_rau_4_mtl: greenPlastic,
  material_name_metall_aluminium939393_gebuerstet_5_mtl: plasticYellow,
  bunny_material: beigeClayMaterial,
};

type ModelConfiguration = {
  url: string;
  view: RawCameraSetup["matrixWorldContent"];
  fov?: number;
  materials: Record<string, OpenPBRMaterial>;
  cameraTargetDistance: number;
  environmentLightConfiguration?: EnvironmentLightConfig;
};

const MODEL_CONFIGURATION = {
  view: [
    -0.2902260614937414, 5.55111512312578e-17, 0.9569581146684689, 0,
    0.27060288181982683, 0.9591865617562106, 0.08206838670951949, 0,
    -0.9179013637535542, 0.28277400825800514, -0.2783809380562283, 0,
    -3.1605996990714327, 0.8489691327469515, -0.6658356804354517, 1,
  ],
  cameraTargetDistance: 5,
  url: "models/optic-effect-demo-scene.glb",
  materials: {
    floor: (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [1.0, 1.0, 1.0];
      m.oSpecularWeight = 0.0;
      return m;
    })(),
    sphere: (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [0.1, 0.1, 0.1];
      m.oSpecularWeight = 1.0;

      return m;
    })(),
    "left-wall": (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [1.0, 0.0, 0.0];
      m.oSpecularWeight = 0.0;
      return m;
    })(),
    "right-wall": (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [1.0, 0.0, 0.0];
      m.oSpecularWeight = 0.0;
      return m;
    })(),
  },
} satisfies ModelConfiguration;

const MODEL_CONFIGURATION1 = {
  view: [
    -0.2902260614937414, 5.55111512312578e-17, 0.9569581146684689, 0,
    0.27060288181982683, 0.9591865617562106, 0.08206838670951949, 0,
    -0.9179013637535542, 0.28277400825800514, -0.2783809380562283, 0,
    -3.1605996990714327, 0.8489691327469515, -0.6658356804354517, 1,
  ],
  cameraTargetDistance: 5,
  url: "models/material-demo-scene.glb",
  materials: {
    floor: (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [1.0, 1.0, 1.0];
      m.oSpecularWeight = 0.0;
      return m;
    })(),
    sphere: (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [0.1, 0.1, 0.1];
      m.oSpecularWeight = 1.0;

      return m;
    })(),
    "left-wall": (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [1.0, 0.0, 0.0];
      m.oSpecularWeight = 0.0;
      return m;
    })(),
    "right-wall": (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [1.0, 0.0, 0.0];
      m.oSpecularWeight = 0.0;
      return m;
    })(),
  },
} satisfies ModelConfiguration;

const MODEL_CONFIGURATION2 = {
  view: [
    0.9396926207859084, 0, -0.3420201433256687, 0, -0.2203032561704394,
    0.7649214009184319, -0.6052782217606094, 0, 0.26161852717499334,
    0.6441236297613865, 0.7187909959242699, 0, 6.531538924716362, 19.5,
    17.948521838355774, 1,
  ],
  cameraTargetDistance: 20,
  url: "models/standard-shader-ball-orig.glb",
  materials: {
    initialShadingGroup: (() => {
      const m = new OpenPBRMaterial();
      m.oBaseColor = [0.8, 0.8, 0.8];
      return m;
    })(),
  },
} satisfies ModelConfiguration;

const MODEL_CONFIGURATION3 = {
  /*[
    -0.9186736378328992, -3.4694469519536134e-18, 0.3950174010735051, 0,
    0.05042967297126754, 0.9918174188143812, 0.1172819501048182, 0,
    -0.391785139119489, 0.1276644341090268, -0.9111565162082436, 0,
    -92.59153816098393, 8.229275754218339, -174.61818921203653, 1,
  ]*/ /*[
    0.9348898557149565, 0, -0.354937963144642, 0, 0.04359232917084678,
    0.992429364980685, 0.1148201391807842, 0, 0.3522508573711748,
    -0.12281675587652569, 0.9278121458340784, 0, 63.44995297630283,
    -44.22427925573443, 209.99999999999994, 1,
  ]*/ /*[
    0.8848578984503438, 3.4694469519536134e-18, -0.46586103029770654, 0,
    0.05721554043046017, 0.992429364980685, 0.10867537649939145, 0,
    0.4623341664676005, -0.12281675587652563, 0.878158962257218, 0,
    18.643087079149865, -26.47294378108665, 73.14454567449334, 1,
  ]*/
  view:
    // color bleeding angle
    [
      0.8848578984503438, 3.4694469519536134e-18, -0.4658610302977065, 0,
      0.05721554043046017, 0.992429364980685, 0.10867537649939148, 0,
      0.46233416646760045, -0.12281675587652566, 0.878158962257218, 0,
      36.036875743252196, -31.093516628749303, 106.1823622816722, 1,
    ],
  cameraTargetDistance: 200,
  url: "models/series-45-nice-back/45-series-cleaned.gltf",
  //url: "https://stuckisimon.github.io/strahl-sample-models/45-series/45-series-cleaned.gltf",
  materials: MATERIAL_MAP,
} satisfies ModelConfiguration;

const MODEL_CONFIGURATION4 = {
  view: [
    -0.8271630755153504, 1.387778780781445e-17, 0.5619619617945562, 0,
    0.11823194300666574, 0.9776172433227516, 0.1740279667492908, 0,
    -0.5493837039418393, 0.2103913628408345, -0.8086488856636859, 0,
    -34.59202033096007, 9.88477047786267, -61.06157562666925, 1,
  ],
  cameraTargetDistance: 100,
  url: "models/metal-wall.glb",
  materials: MATERIAL_MAP,
  environmentLightConfiguration: {
    sky: {
      power: 0.5,
      color: [1.0, 1.0, 1.0],
    },
    sun: {
      power: 1.0,
      angularSize: 35,
      latitude: 40,
      longitude: 160,
      color: [1.0, 1.0, 1.0],
    },
  },
} satisfies ModelConfiguration;

const MODEL_CONFIGURATION5 = {
  view: [
    -0.12972083414957364, 0, 0.9915505560421708, 0, -0.01178931338415953,
    0.999929314122521, -0.0015423515794779758, 0, -0.9914804674210516,
    -0.011889775374860578, -0.12971166471858442, 0, -102.860841569502,
    -8.140322707756264, -2.768869203894816, 1,
  ],
  materials: MATERIAL_MAP,
  url: "models/series-61-rotated/61-serie-edit.gltf",
  cameraTargetDistance: 120,
  environmentLightConfiguration: {
    sky: {
      power: 0.7,
      color: [0.8, 0.8, 1.0],
    },
    sun: {
      power: 1.0,
      angularSize: 30,
      latitude: 140,
      longitude: 310,
      color: [1.0, 1.0, 0.9],
    },
  },
} satisfies ModelConfiguration;

const MODEL_CONFIGURATION6 = {
  view: [
    0.8777699403676876, 8.673617379884032e-19, 0.4790823851770226, 0,
    0.009207871854741938, 0.9998152824140633, -0.0168705704465917, 0,
    -0.4789938902353677, 0.019219808825447805, 0.8776078008232947, 0,
    -64.56343909974215, -3.520641209395011, 160.6180819295023, 1,
  ],
  materials: MATERIAL_MAP,
  url: "models/series-45-raised/45-series.gltf",
  cameraTargetDistance: 120,
} satisfies ModelConfiguration;

const m7red = new OpenPBRMaterial();
m7red.oBaseColor = [0.64, 0.16, 0.1];
m7red.oSpecularWeight = 1.0;
m7red.oSpecularRoughness = 0.3;
m7red.oBaseDiffuseRoughness = 0;

const m7yellow = new OpenPBRMaterial();
m7yellow.oBaseColor = [0.93, 0.84, 0.31];

const m7black = new OpenPBRMaterial();
m7black.oBaseColor = [0.18, 0.18, 0.18];
m7black.oSpecularWeight = 1.0;

const m7metal = new OpenPBRMaterial();
m7metal.oBaseColor = [0.9, 0.9, 0.9];
m7metal.oBaseMetalness = 1.0;

// Visual comparison
const MODEL_CONFIGURATION7 = {
  view: [
    -0.7225167980963385, 2.7755575615628895e-17, 0.6913533658474615, 0,
    0.3268647997748863, 0.8811752561709411, 0.3415985517829365, 0,
    -0.6092034792552791, 0.4727897713699775, -0.6366639246503488, 0,
    -83.96820412623283, 70.07143924538242, -90.78647493857123, 1,
  ],
  fov: 40,
  materials: {
    material_name_kunststoff_schwarz000000_rau_0_mtl: m7black,
    material_name_kunststoff_gelbFCBE37_rau_3_mtl: m7yellow,
    material_name_kunststoff_verkehrsrotB81D12_rau_1_mtl: m7red,
    material_name_metall_stahl737364_abgenutzt_rau_6_mtl: m7metal,
    material_name_metall_stahl909086_abgenutzt_rauer_7_mtl: m7metal,
    material_name_kunststoff_schwarz000000_rau_4_mtl: m7black,
    material_name_kunststoff_schwarz000000_rau_9_mtl: m7black,
    material_name_kunststoff_schwarz000000_rau_13_mtl: m7black,
    material_name_kunststoff_gruen006054_rau_5_mtl: greenPlastic,
    material_name_kunststoff_verkehrsrotB81D12_rau_15_mtl: m7red,
    material_name_kunststoff_weissDCDCDC_transparent_trueb_2_mtl: m7black,
    material_name_kunststoff_weissDCDCDC_transparent_trueb_8_mtl: whitePlastic,
  },
  url: "models/series-45-physical/45-series-physical.glb",
  cameraTargetDistance: 120,
  environmentLightConfiguration: {
    sky: {
      power: 1.0,
      color: [0.8, 0.8, 1.0],
    },
    sun: {
      power: 1.0,
      angularSize: 30,
      latitude: 15,
      longitude: -50,
      color: [1.0, 1.0, 0.9],
    },
  },
} satisfies ModelConfiguration;

const CONFIGURATION_LIST: ModelConfiguration[] = [
  MODEL_CONFIGURATION,
  MODEL_CONFIGURATION1,
  MODEL_CONFIGURATION2,
  MODEL_CONFIGURATION3,
  MODEL_CONFIGURATION4,
  MODEL_CONFIGURATION5,
  MODEL_CONFIGURATION6,
  MODEL_CONFIGURATION7,
];

async function run() {
  const modelConfig = CONFIGURATION_LIST[7];
  const materialMap = modelConfig.materials;

  const model: any = await loadGltf(modelConfig.url);

  // NOTE: This is a hack because on one model, the material name is off
  let alreadyLoadedTheRedOne = false;
  model.scene.traverseVisible((object: any) => {
    if (object.material === undefined) {
      return;
    }
    const materialName = object.material.name;
    if (materialName in materialMap) {
      object.material = materialMap[materialName as keyof typeof materialMap];
      if (
        materialName === "material_name_kunststoff_verkehrsrotB81D12_rau_5_mtl"
      ) {
        if (!alreadyLoadedTheRedOne) {
          object.material = copperMetalMaterial;
        }
        alreadyLoadedTheRedOne = true;
      }
    } else {
      console.log(materialName);
      object.material = defaultBlueMaterial;
    }
    // NOTE: use this to assign same material to all
    // object.material = plainMaterial;
  });

  const destroyController = new AbortController();
  const signal = destroyController.signal;

  try {
    await runPathTracer("render-target", model, {
      clearColor: [1.0, 1.0, 1.0],
      targetSamples: 999,
      enableDenoise: {
        type: "gaussian",
        threshold: 0.09,
      },
      samplesPerIteration: 1,
      maxRayDepth: 5,
      size: 1024,
      viewProjectionConfiguration: {
        matrixWorldContent: modelConfig.view,
        cameraTargetDistance: modelConfig.cameraTargetDistance,
        fov: modelConfig.fov ?? 38.6701655,
      },
      environmentLightConfiguration:
        modelConfig?.environmentLightConfiguration ?? {
          sky: {
            power: 0.5,
            color: [0.8, 0.8, 1.0],
          },
          sun: {
            power: 0.8,
            angularSize: 35,
            latitude: 40,
            longitude: 160,
            color: [1.0, 1.0, 0.9],
          },
        },
      signal,
    });
  } catch (e) {
    console.error("%O", e);
  }
}

run();

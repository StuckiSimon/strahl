import React from "react";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import runPathTracer, {
  OpenPBRMaterial as RawOpenPBRMaterial,
} from "../../../../strahl-lib/dist/strahl";
import styles from "./styles.module.css";
import clsx from "clsx";

// todo: handle types for OpenPBRMaterial
const OpenPBRMaterial = RawOpenPBRMaterial as any;

async function init() {
  const MODEL_URL =
    "https://stuckisimon.github.io/strahl-sample-models/45-series/45-series-cleaned.gltf";

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
  // for 45-cleaned
  redPlasticMaterial.oSpecularWeight = 0.01;

  const metalMaterial = new OpenPBRMaterial();
  metalMaterial.oBaseColor = [0.9, 0.9, 0.9];
  metalMaterial.oBaseMetalness = 1.0;
  metalMaterial.oBaseDiffuseRoughness = 0.5;

  const roughMetalMaterial = new OpenPBRMaterial();
  roughMetalMaterial.oBaseColor = [0.9, 0.9, 0.9];
  //roughMetalMaterial.oBaseMetalness = 0.1;

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
  // for 45-cleaned
  blackPlastic.oSpecularWeight = 0.3;

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
  };

  async function run() {
    const model: any = await loadGltf(MODEL_URL);

    model.scene.traverseVisible((object: any) => {
      if (object.material === undefined) {
        return;
      }
      const materialName = object.material.name;
      if (materialName in MATERIAL_MAP) {
        object.material =
          MATERIAL_MAP[materialName as keyof typeof MATERIAL_MAP];
      } else {
        console.log(materialName);
        object.material = defaultBlueMaterial;
      }
    });

    await runPathTracer("render-target", model);
  }

  run();
}

export default function TracerDemo(): JSX.Element {
  const loadingState = React.useRef(false);
  React.useEffect(() => {
    if (!loadingState.current) {
      loadingState.current = true;
      init();
    }
  }, []);
  return (
    <div className={styles.container}>
      <div className={clsx("container", styles.wrapper)}>
        <canvas width="512" height="512" id="render-target"></canvas>
      </div>
    </div>
  );
}

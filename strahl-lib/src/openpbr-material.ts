import type { Material } from "three";
import { Color } from "./core/types";

export class OpenPBRMaterial {
  oBaseWeight: number = 1.0;
  oBaseColor: Color = [0.8, 0.8, 0.8];
  oBaseMetalness: number = 0.0;
  oBaseDiffuseRoughness: number = 0.0;
  oSpecularWeight: number = 1.0;
  oSpecularColor: Color = [1.0, 1.0, 1.0];
  oSpecularRoughness: number = 0.3;
  oSpecularRoughnessAnisotropy: number = 0.0;
  oSpecularIor: number = 1.5;
  // NOTE: Coat workflow is not yet supported by the renderer
  oCoatWeight: number = 0.0;
  oCoatColor: Color = [1.0, 1.0, 1.0];
  oCoatRoughness: number = 0.0;
  oCoatRoughnessAnisotropy: number = 0.0;
  oCoatIor: number = 1.6;
  oCoatDarkening: number = 1.0;
  oEmissionLuminance: number = 0.0;
  oEmissionColor: Color = [1.0, 1.0, 1.0];
  // NOTE: Thin-film workflow is not yet supported by the renderer
  oThinFilmThickness: number = 0.5;
  oThinFilmIor: number = 1.5;
}

/**
 * Util for use in TypeScript only, once the material is assigned, Three.js will no longer be able to recognize it as a Material
 * @param material
 * @returns
 */
export function asThreeJsMaterial(material: OpenPBRMaterial): Material {
  return material as unknown as Material;
}

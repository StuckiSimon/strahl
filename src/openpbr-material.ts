import { MeshPhysicalMaterial } from "three";

type Color = [number, number, number];

export class OpenPBRMaterial extends MeshPhysicalMaterial {
  oBaseWeight: number = 1.0;
  oBaseColor: Color = [0.8, 0.8, 0.8];
  oBaseDiffuseRoughness: number = 0.0;
  oBaseMetalness: number = 0.0;
  oSpecularWeight: number = 1.0;
  oSpecularColor: Color = [1.0, 1.0, 1.0];
  oSpecularRoughness: number = 0.3;
  oSpecularRoughnessAnisotropy: number = 0.0;
  oSpecularRoughnessRotation: number = 0.0;
  oCoatWeight: number = 0.0;
  oCoatRoughness: number = 0.0;
  oEmissionLuminance: number = 0.0;
  oEmissionColor: Color = [1.0, 1.0, 1.0];
  oThinFilmThickness: number = 0.5;
  oThinFilmIOR: number = 1.5;

  constructor() {
    super();
  }
}

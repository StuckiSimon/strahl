import { InvalidMaterialError } from "../core/exceptions";
import {
  VariableDefinition,
  getSizeAndAlignmentOfUnsizedArrayElement,
  makeStructuredView,
} from "webgpu-utils";
import { OpenPBRMaterial } from "../openpbr-material";
import { Material } from "three";

export function isValidMaterialStructure(
  materials: (Material | Material[])[],
): materials is Material[] {
  return !materials.some((m) => Array.isArray(m));
}

export function generateMaterialBuffer(
  device: GPUDevice,
  materials: Material[],
  materialDefinition: VariableDefinition,
) {
  // CODE#MEMORY-VIEW
  const { size: bytesPerMaterial } =
    getSizeAndAlignmentOfUnsizedArrayElement(materialDefinition);

  const materialDataView = makeStructuredView(
    materialDefinition,
    new ArrayBuffer(bytesPerMaterial * materials.length),
  );

  const materialBuffer = device.createBuffer({
    label: "Material buffer",
    size: bytesPerMaterial * materials.length,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  // CODE#BUFFER-MAPPING
  materialDataView.set(
    materials.map((m) => {
      if (!(m instanceof OpenPBRMaterial)) {
        throw new InvalidMaterialError(m);
      }
      return {
        baseWeight: m.oBaseWeight,
        baseColor: m.oBaseColor,
        baseDiffuseRoughness: m.oBaseDiffuseRoughness,
        baseMetalness: m.oBaseMetalness,
        specularWeight: m.oSpecularWeight,
        specularColor: m.oSpecularColor,
        specularRoughness: m.oSpecularRoughness,
        specularAnisotropy: m.oSpecularRoughnessAnisotropy,
        specularIor: m.oSpecularIor,
        coatWeight: m.oCoatWeight,
        coatColor: m.oCoatColor,
        coatRoughness: m.oCoatRoughness,
        coatRoughnessAnisotropy: m.oCoatRoughnessAnisotropy,
        coatIor: m.oCoatIor,
        coatDarkening: m.oCoatDarkening,
        emissionLuminance: m.oEmissionLuminance,
        emissionColor: m.oEmissionColor,
        thinFilmThickness: m.oThinFilmThickness,
        thinFilmIor: m.oThinFilmIor,
      };
    }),
  );

  const materialMapped = materialBuffer.getMappedRange();
  const materialMappedData = new Uint8Array(materialMapped);
  materialMappedData.set(new Uint8Array(materialDataView.arrayBuffer));
  materialBuffer.unmap();

  return materialBuffer;
}

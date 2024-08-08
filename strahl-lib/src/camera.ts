import { Camera } from "three";
import { isNil } from "./is-nil";

export type CustomCameraSetup = {
  camera: Camera;
  controls: {
    addEventListener: (event: "change", listener: () => void) => void;
  };
};

export type ViewProjectionConfiguration =
  | CustomCameraSetup
  | {
      // todo: add more precise type
      matrixWorldContent: number[];
      fov: number;
      cameraTargetDistance: number;
    };

export function isCustomCameraSetup(
  viewProjectionConfiguration: ViewProjectionConfiguration,
): viewProjectionConfiguration is CustomCameraSetup {
  return !isNil((viewProjectionConfiguration as CustomCameraSetup).camera);
}

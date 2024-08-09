import {
  Camera,
  EventDispatcher,
  Matrix4,
  PerspectiveCamera,
  Vector3,
} from "three";
import { isNil } from "./is-nil";
import { OrbitControls } from "three/examples/jsm/Addons.js";

export type CustomCameraSetup = {
  camera: Camera;
  controls?: EventDispatcher;
};

type RawCameraSetup = {
  // todo: add more precise type
  matrixWorldContent: number[];
  fov: number;
  cameraTargetDistance: number;
};

export type ViewProjectionConfiguration = CustomCameraSetup | RawCameraSetup;

export function isCustomCameraSetup(
  viewProjectionConfiguration: ViewProjectionConfiguration,
): viewProjectionConfiguration is CustomCameraSetup {
  return !isNil((viewProjectionConfiguration as CustomCameraSetup).camera);
}

export function makeRawCameraSetup(
  rawCameraSetup: RawCameraSetup,
  canvas: HTMLCanvasElement,
): CustomCameraSetup {
  const matrixWorld = new Matrix4();

  matrixWorld.fromArray(rawCameraSetup.matrixWorldContent);

  // todo: externalize aspect ratio
  const camera = new PerspectiveCamera(rawCameraSetup.fov, 1, 0.01, 1000);
  const internalControls = new OrbitControls(camera, canvas);
  camera.applyMatrix4(matrixWorld);

  const dir = new Vector3();
  camera.getWorldDirection(dir);
  const camTarget = camera.position
    .clone()
    .addScaledVector(dir, rawCameraSetup.cameraTargetDistance);
  internalControls.target.copy(camTarget);

  internalControls.update();

  return {
    camera,
    controls: internalControls,
  };
}
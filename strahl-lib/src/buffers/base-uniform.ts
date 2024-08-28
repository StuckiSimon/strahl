import { Matrix } from "../camera";
import { Color, Vec3 } from "../core/types";
import { EnvironmentLightConfig } from "../environment-light";

export function getBaseUniformData(
  invProjectionMatrix: Matrix,
  matrixWorld: Matrix,
  invModelMatrix: Matrix,
  currentSample: number,
  samplesPerIteration: number,
  sunDirection: Vec3,
  environmentLightConfiguration: EnvironmentLightConfig,
  clearColor: Color | false,
  maxRayDepth: number,
  objectDefinitionLength: number,
) {
  return {
    invProjectionMatrix: invProjectionMatrix,
    cameraWorldMatrix: matrixWorld,
    invModelMatrix,
    seedOffset: Math.random() * Number.MAX_SAFE_INTEGER,
    priorSamples: currentSample,
    samplesPerPixel: samplesPerIteration,
    sunDirection,
    skyPower: environmentLightConfiguration.sky.power,
    skyColor: environmentLightConfiguration.sky.color,
    sunPower: Math.pow(10, environmentLightConfiguration.sun.power),
    sunAngularSize: environmentLightConfiguration.sun.angularSize,
    sunColor: environmentLightConfiguration.sun.color,
    clearColor: clearColor === false ? [0, 0, 0] : clearColor,
    enableClearColor: clearColor === false ? 0 : 1,
    maxRayDepth,
    objectDefinitionLength: objectDefinitionLength,
  };
}

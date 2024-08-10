import { Color, Vec3 } from "./core/types.ts";

/**
 * Configuration for the environment light consisting of ambient sky light and directional sun light.
 */
export type EnvironmentLightConfig = {
  /**
   * Configuration for the ambient sky light.
   */
  sky: {
    power: number;
    color: Color;
  };
  /**
   * Configuration for the directional sun light.
   */
  sun: {
    power: number;
    angularSize: number;
    latitude: number;
    longitude: number;
    color: Color;
  };
};

export function defaultEnvironmentLightConfig(): EnvironmentLightConfig {
  return {
    sky: {
      power: 1,
      color: [1, 1, 1],
    },
    sun: {
      power: 0.5,
      angularSize: 40,
      latitude: 45,
      longitude: 180,
      color: [1, 1, 1],
    },
  };
}

export function getSunDirection(
  sunConfig: EnvironmentLightConfig["sun"],
): Vec3 {
  const { latitude, longitude } = sunConfig;
  const latTheta = ((90.0 - latitude) * Math.PI) / 180.0;
  const lonPhi = (longitude * Math.PI) / 180.0;
  const cosTheta = Math.cos(latTheta);
  const sinTheta = Math.sin(latTheta);
  const cosPhi = Math.cos(lonPhi);
  const sinPhi = Math.sin(lonPhi);
  const x = sinTheta * cosPhi;
  const z = sinTheta * sinPhi;
  const y = cosTheta;
  return [x, y, z];
}

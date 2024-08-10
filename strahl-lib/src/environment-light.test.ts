import { describe, expect, it } from "vitest";
import { EnvironmentLightConfig, getSunDirection } from "./environment-light";

describe("environment-light", () => {
  describe("getSunDirection", () => {
    it("should return the sun direction vector", () => {
      const sunConfig = {
        latitude: 10,
        longitude: 50,
      } satisfies Partial<EnvironmentLightConfig["sun"]>;

      const result = getSunDirection(
        sunConfig as EnvironmentLightConfig["sun"],
      );

      expect(result).toEqual([
        0.6330222215594891, 0.17364817766693041, 0.7544065067354889,
      ]);
    });
  });
});

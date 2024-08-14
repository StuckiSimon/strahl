import { describe, expect, it } from "vitest";
import { convertHexToRGB } from "./hex-to-rgb";

describe("hexToRgb", () => {
  it("should convert hex to rgb", () => {
    const hex = "#ff0055";
    const rgb = convertHexToRGB(hex);
    expect(rgb).toEqual([1, 0, 0.3333333333333333]);
  });

  it("should accept hex string without hash", () => {
    const hex = "ff0055";
    const rgb = convertHexToRGB(hex);
    expect(rgb).toEqual([1, 0, 0.3333333333333333]);
  });
});

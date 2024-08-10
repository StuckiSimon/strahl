import { describe, expect, it } from "vitest";
import { isNil } from "./is-nil.ts";

describe("isNil", () => {
  it("should return true for null", () => {
    expect(isNil(null)).toBe(true);
  });

  it("should return true for undefined", () => {
    expect(isNil(undefined)).toBe(true);
  });

  it("should return false for a value", () => {
    expect(isNil(1)).toBe(false);
  });
});

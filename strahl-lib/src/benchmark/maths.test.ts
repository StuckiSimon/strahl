import { describe, expect, it } from "vitest";
import {
  getConfidenceInterval,
  getSampleMean,
  getStandardDeviation,
} from "./maths";

describe("maths", () => {
  describe("getStandardDeviation", () => {
    it("should return 0 when list is empty", () => {
      expect(getStandardDeviation([])).toBe(0);
    });

    it("should return the standard deviation of a list of numbers", () => {
      const list = [2, 4, 4, 4, 5, 5, 7, 9];
      expect(getStandardDeviation(list)).toBe(2);
    });
  });

  describe("getSampleMean", () => {
    it("should return 0 when list is empty", () => {
      expect(getSampleMean([])).toBe(0);
    });

    it("should return the mean of a list of numbers", () => {
      const list = [1, 2, 4, 5];
      expect(getSampleMean(list)).toBe(3);
    });
  });

  describe("confidenceInterval", () => {
    it("should return the correct confidence interval", () => {
      const xBar = 3.5;
      const s = 2.5;
      const n = 100;
      const z = 1.96; // 95% confidence interval

      const { lowerBound, upperBound, marginOfError } = getConfidenceInterval(
        xBar,
        s,
        n,
        z,
      );

      expect(lowerBound).toBeCloseTo(3.5 - 1.96 * (2.5 / Math.sqrt(100)), 2);
      expect(upperBound).toBeCloseTo(3.5 + 1.96 * (2.5 / Math.sqrt(100)), 2);
      expect(marginOfError).toBeCloseTo(1.96 * (2.5 / Math.sqrt(100)), 2);
    });
  });
});

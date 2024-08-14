import { Color } from "../core/types";

function parseHex(hex: string): number {
  return parseInt(hex, 16);
}

function convertEightBitToNormalized(value: number): number {
  return value / 255;
}

/**
 * Convert a hex color string to a normalized color array for use within strahl.
 * @param hex The hex color string (e.g. #ff0055)
 * @returns The color in normalized format (0-1)
 */
export function convertHexToRGB(hex: string): Color {
  const o = hex[0] === "#" ? 1 : 0;
  return [
    convertEightBitToNormalized(parseHex(hex.slice(o, o + 2))),
    convertEightBitToNormalized(parseHex(hex.slice(o + 2, o + 4))),
    convertEightBitToNormalized(parseHex(hex.slice(o + 4, o + 6))),
  ];
}

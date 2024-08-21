import { ScreenshotCaptureError } from "../core/exceptions";
import { isNil } from "./is-nil";

/**
 * Captures a screenshot of the canvas element and downloads it as a PNG file.
 *
 * @param canvas DOM node of the canvas element
 */
export function captureCanvasScreenshot(canvas: HTMLCanvasElement) {
  canvas.toBlob(
    (blob) => {
      if (isNil(blob)) {
        throw new ScreenshotCaptureError("Blob is null.");
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "screenshot.png";
      a.click();
      URL.revokeObjectURL(url);
    },
    "image/png",
    1,
  );
}

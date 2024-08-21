export class WebGPUNotSupportedError extends Error {
  constructor(techReason: string) {
    super(
      `[WebGPUNotSupportedError] WebGPU is not supported in this environment. Make sure to use a browser and device that supports WebGPU. (Reason: ${techReason})`,
    );
  }
}

export class CanvasReferenceError extends Error {
  constructor() {
    super(
      "[CanvasReferenceError] Canvas reference is not set. Make sure to set the canvas to a DOM node before calling any methods.",
    );
  }
}

export class SignalAlreadyAbortedError extends Error {
  constructor() {
    super(
      "[SignalAlreadyAbortedError] Signal is already aborted. Make sure to check the aborted property before calling any methods.",
    );
  }
}

export class InvalidMaterialGroupError extends Error {
  constructor(public readonly invalidMaterialGroup: unknown) {
    super(
      "[InvalidMaterialGroupError] Material group is invalid. Make sure that all objects have an associated material index.",
    );
  }
}

export class InvalidMaterialError extends Error {
  constructor(public readonly invalidMaterial: unknown) {
    super(
      "[InvalidMaterialError] Material is invalid. Make sure to configure all materials to be instances of OpenPBRMaterial.",
    );
  }
}

export class InternalError extends Error {
  constructor(techReason: string) {
    super(
      `[InternalError] Library error occurred, please get in contact. (Reason: ${techReason})`,
    );
  }
}

export class ScreenshotCaptureError extends Error {
  constructor(techReason: string) {
    super(
      `[ScreenshotCaptureError] Screenshot capture failed. (Reason: ${techReason})`,
    );
  }
}

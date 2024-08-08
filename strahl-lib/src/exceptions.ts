export class WebGPUNotSupportedError extends Error {
  constructor(techReason: string) {
    super(
      `WebGPU is not supported in this environment. Make sure to use a browser and device that supports WebGPU. (Reason: ${techReason})`,
    );
  }
}

export class CanvasReferenceError extends Error {
  constructor() {
    super(
      "Canvas reference is not set. Make sure to set the canvas to a DOM node before calling any methods.",
    );
  }
}

export class SignalAlreadyAbortedError extends Error {
  constructor() {
    super(
      "Signal is already aborted. Make sure to check the aborted property before calling any methods.",
    );
  }
}

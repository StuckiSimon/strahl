# Exception Handling

`strahl` provides parseable exceptions to help debugging failures in the program. For ease-of-use, the name of the exception is written in brackets (`[ExceptionName] …`).

:::tip
Certain exceptions have more detailed information attached as properties, use `console.error("%O", e);` to log the exception object in full detail.
:::

## `WebGPUNotSupportedError`

WebGPU adapter could not be initialized. This is usually due to a lack of support in the current environment. See [Intro](../intro.md#browser-support) for information about browser support and how to check this.

## `CanvasReferenceError`

Make sure the canvas is actually present in the DOM. To check this, you may try something like `document.getElementById(id)` before initializing the path tracer.

## `SignalAlreadyAbortedError`

See [Destruction](../advanced/destruction.md).

## `InvalidMaterialGroupError`

This indicates a bad setup in the object. One part of the model does not have an associated material which is invalid. Make sure that all objects have an associated material index.

## `InvalidMaterialError`

Make sure that all materials of your object are instances of strahl's `OpenPBRMaterial`. This is required to have consistent OpenPBR parameters. See [Material Mapping](../techniques/material-mapping.md) on how you can set this up, or [Material](../material/) on the parameter set.

## `InternalError`

An error within `strahl` has occurred, please reach out with a detailed error report with a reproduction of the issue. These are generally non-recoverable.

## `ScreenshotCaptureError`

An error occurred while capturing a canvas screenshot. Check the tech reason for detailed information.

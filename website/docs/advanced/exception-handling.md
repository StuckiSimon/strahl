# Exception Handling

`strahl` provides parseable exceptions to help debugging failures in the program. For ease-of-use, the name of the exception is written in brackets (`[ExceptionName] …`).

:::tip
Certain exceptions have more detailed information attached as properties, use `console.error("%O", e);` to log the exception object in full detail.
:::

## `WebGPUNotSupportedError`

WebGPU adapter could not be initialized. This is usually due to a lack of support in the current environment. See [Intro](../intro.md#browser-support) for information about browser support and how to check this.

## `InternalError`

An error within `strahl` has occurred, please reach out with a detailed error report with a reproduction of the issue. These are generally non-recoverable.

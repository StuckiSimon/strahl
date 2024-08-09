# Destruction

`strahl` sets up all required data structures for you, but by default it won't clean them up. This can lead to a memory leak. Therefore, it should always be destroyed after it has finished processing or the page is becoming idle. Note that this will not destroy the canvas, therefore it is fine to do even if the scene is still rendered.

:::warning
Make sure to not start path tracing with an already aborted signal.
:::

## AbortSignal

`AbortSignal` is a relatively new web API supported by all major browsers. It allows for the abortion of an operation. For details on the API, please refer to [AbortController on developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/API/AbortController).

## Usage in `strahl`

```js title="strahlConfiguration.js"
const destroyController = new AbortController();
const signal = destroyController.signal;

runPathTracer(target, model, {
  signal,
});
```

## Invoke Destruction

In order to cancel `strahl`, invoke it as:

```js
destroyController.abort();
```

:::warning
This action is irreversible, you need to setup a new instance when doing a new rendering.
:::

## API Design Consideration

While it is more frequent to use a return type to cancel operations, signals have the unique advantage of being usable at all stages of the tracer even before the initial return. Their availability as part of the platform makes them easy to integrate and increases the likelihood of developers being familiar with them.

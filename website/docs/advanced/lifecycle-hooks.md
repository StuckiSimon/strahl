# Lifecycle Hooks

In order to be notified about the state of the path tracer, lifecycle hooks are available. They are handled as event listeners and will be called with information about the current state.

## `onSampleStart`

To be notified before a sample start, attach an `onSampleStart` listener.

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  onSampleStart: ({ cameraPosition }) => {
    console.log(cameraPosition);
  },
});
```

- `cameraPosition` — corresponds to `matrixWorldContent` as configured in [View Projection](./view-projection.md).

## `onSamplingFinished`

To be notified after all samples were taken, attach an `onSamplingFinished` listener.

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  onSamplingFinished: ({
    bvhBuildTime,
    fullRenderLoopTime,
    allRenderTime: renderAgg,
    renderTimes: renderTimes,
  }) => {
    // process
  },
});
```

- `bvhBuildTime` — defines how long it took to build the BVH on the CPU
- `fullRenderLoopTime` — measures wall-clock time of the render loop
- `allRenderTime` — sum of all render times
- `renderTimes` — array of render time measurements. This measure how long the rendering process took per sample.

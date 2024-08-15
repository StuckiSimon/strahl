# Sampling

Sampling determines how many rays are cast per pixel and how deep the rays are sampled.

## Configuration

There are three independent variables to steer sampling.

### `targetSamples`

Determines how many samples should be done. Each iteration is visualized.

Commonly set between 100 - 10'000.

### `samplesPerIteration`

How many samples should be done within each iteration.

`samplesPerIteration * (targetSamples + 1) = total samples`

Commonly set to 1 - 5.

### `maxRayDepth`

How many bounces a given ray should do before termination.
Commonly set to 3 - 20.

## In Code

:::warning
Setting `samplesPerIteration` or `maxRayDepth` to high values may lead to a GPU crash
:::

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  targetSamples: 100,
  samplesPerIteration: 1,
  maxRayDepth: 2,
});
```

---
sidebar_position: 2
---

# Getting Started

This is a minimal guide on how to setup `strahl` in 5 minutes.

## Installation

The library is available in the npm registry. To install it using npm, run:

```
npm install --save strahl
```

## Setup

### Define canvas

```html title="index.html"
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  </head>
  <body>
    <canvas width="512" height="512" id="render-target"></canvas>
  </body>
</html>
```

:::warning
Make sure that the `canvas` is present in the `DOM` before calling `strahl`.
:::

### Prepare 3D data

Load 3D models using `three` loaders.

```js title="loadModel.js"
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const gltfLoader = new GLTFLoader();

async function loadGltf(url) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

const model = await loadGltf(MODEL_URL);
```

:::tip
If you don't have a model at hand, take "https://stuckisimon.github.io/strahl-sample-models/45-series/45-series-cleaned.gltf". Please adhere to licenses associated with the model as described in [Sample Models](https://stuckisimon.github.io/strahl-sample-models/)
:::

Make sure to assign `OpenPBRMaterial` to all materials within the scene.

```js title="materialMapper.js"
const MATERIAL_MAP = {
  'name': new OpenPBRMaterial()
}

model.scene.traverseVisible((object: any) => {
  if (object.material === undefined) {
    return;
  }
  const materialName = object.material.name;
  if (materialName in MATERIAL_MAP) {
    object.material =
      MATERIAL_MAP[materialName as keyof typeof MATERIAL_MAP];
  } else {
    console.log('unknown material', materialName);
    object.material = defaultBlueMaterial;
  }
});
```

That's it! Now you're ready to start path tracing.

### Initialize `strahl`

```js
await runPathTracer("render-target", model);
```

:::info
`runPathTracer` accepts a third parameter for additional configuration. See the advanced guides for more information.
:::

Et voilà! You should now see the path tracer rendering the scene.

### Help me! It doesn't work

Feel free to reach out, happy to help you out and potentially improve guidance or fix bugs.

### What's next?

Check out the [advanced guides](/docs/category/advanced) for more information on how to configure the path tracer.

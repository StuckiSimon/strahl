---
sidebar_position: 2
---

# Getting Started

This is the full guide on how to setup `strahl` in a new project.

## Project Setup

While `strahl` can be set up in a variety of projects, we'll take [Vite](https://vitejs.dev/) as a basis. If you already have a project setup, you can skip to [Strahl Installation](#strahl-installation).

Open a terminal in your project folder and run:

```
npm create vite@latest
```

In the wizard, choose a project name (`your-name`) and set the follow-up prompts to `Vanilla` for framework and `JavaScript` for variant respectively.

## Setup Verification

Now is a good time to open the project in the IDE of your choice.

```
cd your-name
npm install
npm run dev
```

You should have a basic website set up with Vite running on `http://localhost:5173/` (port may vary). Now we're ready to install `strahl`.

## Strahl Installation

The library is available in the npm registry. To install it, run:

```
npm install --save strahl three
npm i --save-dev @types/three
```

We'll use `Three.js` for the scene setup.

## Strahl Setup

To get the first rendering, we need to configure `strahl`.

### Change `index.html`

Let's replace the app div with our canvas.

```html title="index.html"
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vite App</title>
  </head>
  <body>
    <!-- diff-remove -->
    <div id="app"></div>
    <!-- diff-add -->
    <canvas width="512" height="512" id="render-target"></canvas>
    <script type="module" src="/main.js"></script>
  </body>
</html>
```

### Change `main.js`

Let's remove the demo code from `main.js`. Delete all the contents within `main.js`. You can also remove the corresponding resources which are imported.

:::note
The file should be empty.
:::

### Prepare 3D data

Before we can create renderings, we need some 3D data. We'll start by using `strahl` demo scenes.

:::warning
Please adhere to licenses associated with the model as described in [Sample Models](https://stuckisimon.github.io/strahl-sample-models/)
:::

Load 3D models using `three` loaders.

```ts title="main.js"
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const gltfLoader = new GLTFLoader();

async function loadGltf(url) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

const MODEL_URL =
  "https://stuckisimon.github.io/strahl-sample-models/sphere-plane/demo1.glb";

const model = await loadGltf(MODEL_URL);

console.log(model);
```

You should see the log of the model in the console (DevTools shortcut is often `F12`).

#### Assign material

Now we need to set a material onto the objects.

```ts title="main.js"
// diff-add
import { OpenPBRMaterial } from "strahl";
…
const model = await loadGltf(MODEL_URL);

// diff-remove
console.log(model);
// diff-add

// diff-add
model.scene.traverseVisible((object) => {
// diff-add
  if (object.material === undefined) {
// diff-add
    return;
// diff-add
  }
// diff-add
  object.material = new OpenPBRMaterial();
// diff-add
});
// diff-add

// diff-add
console.log(model);
```

That's it! Now you're ready to start path tracing.

### Initialize `strahl`

To start tracing, we'll have to configure the model and the camera settings.

```ts title="main.js"
// diff-remove
import { OpenPBRMaterial } from "strahl";
// diff-add
import { OpenPBRMaterial, runPathTracer } from "strahl";

…

// diff-remove
console.log(model);
// diff-add
await runPathTracer("render-target", model, {
  // diff-add
  viewProjectionConfiguration: {
    // diff-add
    matrixWorldContent: [
      // diff-add
      -0.45178184301411944, 4.163336342344336e-17, 0.8921284472108064, 0,
      // diff-add
      0.18290622579667423, 0.9787573022265018, 0.09262535237781978, 0,
      // diff-add
      -0.8731772322315672, 0.20502229961225985, -0.44218477786341664, 0,
      // diff-add
      -3.67881274400709, 0.6362064645963488, -1.879628578827991, 1,
      // diff-add
    ],
    // diff-add
    fov: 38,
    // diff-add
    aspect: 1,
    // diff-add
    cameraTargetDistance: 4,
    // diff-add
  },
  // diff-add
});
```

Et voilà! You should now see the path tracer rendering the scene.

## Help me! It doesn't work

Feel free to reach out, happy to help you out and potentially improve guidance or fix bugs.

## What's next?

There's lots more to try out, you can:

- do [material configuration](./material/)
- apply [material mapping](./techniques/material-mapping) to have multiple different materials in the same scene
- try different [demo models](https://github.com/StuckiSimon/strahl-sample-models)
- check out the [advanced guides](/docs/category/advanced) for more information on how to configure the path tracer.

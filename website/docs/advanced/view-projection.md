# View Projection

View projection is about setting up the virtual camera and getting the right shot. To date, `strahl` only supports perspective view projection as encountered in real life.

In order to setup view projection, a camera is required. If available, the associated controls must also be configured in order for the path tracer to register on changes and update the view as the controls are updated.

## Basic Configuration

`strahl` offers basic camera configuration which is suitable for minimal cases. It permits configuring the camera viewport and some basic input.

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  viewProjectionConfiguration: {
    matrixWorldContent: [
      -0.2902260614937414, 5.55111512312578e-17, 0.9569581146684689, 0,
      0.27060288181982683, 0.9591865617562106, 0.08206838670951949, 0,
      -0.9179013637535542, 0.28277400825800514, -0.2783809380562283, 0,
      -3.1605996990714327, 0.8489691327469515, -0.6658356804354517, 1,
    ],
    fov: 1.0,
    aspect: 0.1,
    cameraTargetDistance: 10,
  },
});
```

- `matrixWorldContent` — rotation and position of the camera as `4x4` matrix
- `fov` — field of view of the camera in degrees
- `aspect` — aspect ratio
- `cameraTargetDistance` — the distance to the center that the `OrbitControls` should orbit around

:::tip
To get the `matrixWorldContent` of the current scene, you can use the corresponding lifecycle hook as described in [Lifecycle Hooks](./lifecycle-hooks.md).
:::

## Full Configuration

Alternatively, you can also set up custom camera and control instances. For details on how to setup the `PerspectiveCamera`, see [Three.js PerspectiveCamera](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera).

An example configuration may look like:

```js title="cameraConfiguration.js"
import { PerspectiveCamera } from "three";

const camera = new PerspectiveCamera(38, 1, 0.01, 1000);
```

### (Optional) Camera Controls

`strahl` supports arbitrary controls, to set up, use for example:

```js title="controlsConfiguration.js"
import { OrbitControls } from "three/examples/jsm/Addons.js";

const controls = new OrbitControls(camera, canvas);
```

### Tying it to `strahl`

The camera configuration can be passed to strahl as an optional parameter, to do so:

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  viewProjectionConfiguration: {
    camera,
    controls,
  },
});
```

:::note
`strahl` will attach itself to the controls to be notified of changes.
:::

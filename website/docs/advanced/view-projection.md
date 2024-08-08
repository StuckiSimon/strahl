# View Projection

View projection is about setting up the virtual camera and getting the right shot. To date, `strahl` only supports perspective view projection as encountered in real life.

In order to setup view projection, a camera is required. If available, the associated controls must also be configured in order for the path tracer to register on changes and update the view as the controls are updated.

## Configuration

For details on how to setup the `PerspectiveCamera`, see [Three.js PerspectiveCamera](https://threejs.org/docs/index.html#api/en/cameras/PerspectiveCamera).

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

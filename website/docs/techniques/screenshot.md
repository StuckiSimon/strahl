# Screenshot

In order to download a picture of the rendering, `strahl` offers a dedicated screenshot utility.

## Configuration

```js title="screenshot.js"
import { captureCanvasScreenshot } from "strahl";

…

// this must be a canvas instance
const canvas = document.getElementById("render-target")
captureCanvasScreenshot(canvas)
```

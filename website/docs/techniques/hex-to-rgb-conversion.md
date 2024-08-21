# Hex to RGB Conversion

Colors are frequently encoded as hex, e.g. `#f0f0f0`. Therefore, `strahl` offers a utility to convert this to the normalized RGB color space.

## Configuration

```js title="convertHexToRGB.js"
import { convertHexToRGB } from "strahl";

…

const hex = "#ff0055";
const rgb = convertHexToRGB(hex);
```

---
sidebar_position: 1
---

# `strahl` Documentation

If you want to get started as quick as possible, take a look at the [Getting Started](/docs/tutorial-basic) tutorial. If you're interested in some more information about what `strahl` is, check out [information](/docs/information) first.

## Quick Links

- [ACM short paper](https://doi.org/10.1145/3665318.3677158) — summary of the work
- [report](https://github.com/StuckiSimon/strahl/blob/report/report.pdf) — master thesis report with full details
- [npm package](https://www.npmjs.com/package/strahl) — installable `strahl` package

## Documentation Structure

- [Material](./category/material/) — includes information about how to configure OpenPBR parameters
- [Advanced](./category/advanced/) — practical guides on how to setup the renderer
- [Techniques](./category/techniques/) — common techniques which can be used in combination with `strahl`
- [Library Types](./api/) — TypeScript type documentation

## Browser Support

Strahl is written in WebGPU. WebGPU is a new web API which is not yet supported in all browsers. Check [caniuse.com](https://caniuse.com/webgpu) for details about the support. To check if your current browser supports WebGPU, head over to [webgpureport.org](https://webgpureport.org/).

WebGPU is restricted to secure contexts. This includes `localhost`, but otherwise requires use of `HTTPS`. See [Secure Context on developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) for more information.

---
sidebar_position: 1
---

# `strahl` Documentation

:::danger[Disclaimer]
The tutorial is not yet finished and will be following in the upcoming weeks.
:::

## Where to find what

This is the minimal getting started documentation focusing on hands-on technical documentation. It is suitable if you are interested in creating renderings with `strahl`. However, if you're more interested in how it could be used, what the design decisions were and the internal working of the path tracer, there is a report with more extensive information available on the [Github](https://github.com/StuckiSimon/strahl) repository. If you're unsure about whether this is interesting to you, there is a short-paper which gives a brief introduction into the topic available on [arXiv](https://arxiv.org/abs/2407.19977).

## Browser Support

Strahl is written in WebGPU. WebGPU is a new web API which is not yet supported in all browsers. Check [caniuse.com](https://caniuse.com/webgpu) for details about the support. To check if your current browser supports WebGPU, head over to [webgpureport.org](https://webgpureport.org/).

WebGPU is restricted to secure contexts. This includes `localhost`, but otherwise requires use of `HTTPS`. See [Secure Context on developer.mozilla.org](https://developer.mozilla.org/en-US/docs/Web/Security/Secure_Contexts) for more information.

---
sidebar_position: 3
---

# Information

**tl;dr** `strahl` is a WebGPU Path Tracer for the web.

strahl is a path tracing library for web applications. Path tracing is a technique based on ray tracing to render 3D models to images with realistic reflections and ambient occlusion without the need for pregenerated artifacts. strahl leverages WebGPU for optimal performance and is based on the OpenPBR surface shading model.

:::note
The [master thesis report](https://github.com/StuckiSimon/strahl/blob/report/report.pdf) contains a thorough introduction into the use cases, ray tracing and path tracing and a comparison to rasterization. This serves as a very brief summary.
:::

## Real-time Rendering with Rasterization

Rasterization is the predominant technique to perform real-time rendering on the web. If you're familiar with [Three.js](https://threejs.org/) or [Babylon.js](https://www.babylonjs.com/) you most likely already used rasterization.

However, rasterization has limitations when it comes to implementing effects such as shadow casting, reflections, color bleeding, and ambient occlusion. These are subtle effects which are crucial for high-fidelity rendering.

## Ray Tracing

Ray tracing inherently addresses these limitations but at the cost of increased rendering time. `strahl` is a library which leverages ray tracing to generate interactive 3D renderings in the browser.

:::note
Path tracing is a specific implementation of ray tracing and describes the chosen algorithm.
:::

## Use Case

`strahl` is designed to be used to for use cases where near real-time rendering performance is sufficient, pregenerated artifacts are hard to obtain, and visual fidelity is important. The demo from the start page is one such example of a complex CAD model which consists of one million triangles and is derived from production CAD data.

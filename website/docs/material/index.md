---
sidebar_position: 0
---

# Intro Material

`strahl` leverages the [OpenPBR surface shading model](https://github.com/AcademySoftwareFoundation/OpenPBR) to achieve physically-based rendering.

## What is Physically-based Rendering?

Physically-based rendering, or short PBR, is an approach that seeks to render models in a way that resembles physical laws of reality. The goal is to have physically accurate renderings for a wide range of environments.

## Material Definition

These tutorials establish an understanding of the parameters available to adjust the material appearance. All of these properties can be set on the `OpenPBRMaterial`, for example, to set the `baseColor`, do:

```js
const material = new OpenPBRMaterial();
material.oBaseColor = [0.4, 1.0, 0.0];
```

:::tip
Use code completion to aid in finding parameters.
:::

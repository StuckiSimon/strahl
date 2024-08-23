# Denoising

Denoising can be used as a final step to reduce undesired artifacts in the rendering.

## Configuration

There are two independent denoising options available.

### Denoise filtering

Simple denoise filtering using circular Gaussian kernel. To enable the default settings, use:

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  enableDenoise: true,
});
```

If you want more control, pass a configuration object instead:

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  enableDenoise: {
    type: "gaussian",
    sigma: 4.0,
    kSigma: 1.0,
    threshold: 0.1,
  },
});
```

### OIDN

[Open Image Denoise](https://github.com/RenderKit/oidn) is a denoise library which uses machine learning techniques to reduce noise. To configure it, you must provide a URL to a weight file. One option is: [oidn-weights on GitHub.com](https://github.com/RenderKit/oidn-weights/blob/master/rt_hdr_alb_nrm.tza).

:::warning
OIDN may introduce undesired artifacts
:::

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  enableDenoise: {
    type: "oidn",
    url: "./oidn-weights/rt_hdr_alb_nrm.tza",
  },
});
```
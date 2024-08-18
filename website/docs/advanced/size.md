# Size

Configure the size of the rendering.

:::tip
The size of the rendering does not have to match the size of the canvas. Using higher rendering resolutions can be used for higher pixel density.
:::

## Configuration

There are two ways to configure it, either as a square or as `width` and `height` object.

### Square configuration

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  size: 512,
});
```

### Custom Size

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  size: {
    width: 1024,
    height: 512,
  },
});
```

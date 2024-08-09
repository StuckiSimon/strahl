# Clear Color

The clear color is used as the background of the rendering. Per default, the background is rendered as white, as is common for product renderings. However, you can either disable it or add a custom clear color.

## Configuration

To disable clear color, use:

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  clearColor: null,
});
```

This will render a rudimentary skybox.

To configure a custom `clearColor`, pass it as:

```js title="strahlConfiguration.js"
runPathTracer(target, model, {
  clearColor: [1.0, 1.0, 0.8],
});
```

:::info
The `clearColor` has no impact on the rendering, it does not affect the lighting situation. If you want to change the lighting, you may use [environment light](./environment-light.md).
:::

# Material Mapping

Assigning materials is crucial for rendering. This chapter explains how to map materials onto a model using material mapping.

:::warning
All materials of an object need to be `OpenPBRMaterial` instances.
:::

## Configuration

One can do so by defining a `Map` which contains for each material name the corresponding `OpenPBR` configuration.

```js title="materialMapper.js"
import { OpenPBRMaterial } from "strahl";

…

const MATERIAL_MAP = {
  name: new OpenPBRMaterial(),
};

const defaultBlueMaterial = new OpenPBRMaterial();
defaultBlueMaterial.oBaseColor = [0.0, 0.9, 1.0];

model.scene.traverseVisible((object) => {
  if (object.material === undefined) {
    return;
  }
  const materialName = object.material.name;
  if (materialName in MATERIAL_MAP) {
    object.material = MATERIAL_MAP[materialName];
  } else {
    // this will give you the name of the material to configure in the map
    console.log("unknown material", materialName);
    object.material = defaultBlueMaterial;
  }
});
```

import React from "react";
import { OpenPBRMaterial, runPathTracer } from "strahl";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

let modelCache = new Map<string, any>();

const gltfLoader = new GLTFLoader();

async function loadGltf(url: string) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

const defaultBlueMaterial = new OpenPBRMaterial();
defaultBlueMaterial.oBaseColor = [0.0, 0.9, 1.0];

async function init(
  modelUrl: string,
  materialMap: Record<string, OpenPBRMaterial>,
  options: Parameters<typeof runPathTracer>[2],
) {
  let model = modelCache.get(modelUrl);
  if (model === undefined) {
    model = await loadGltf(modelUrl);
    modelCache.set(modelUrl, model);
  }

  model.scene.traverseVisible((object: any) => {
    if (object.material === undefined) {
      return;
    }
    const materialName = object.material.name;
    if (materialName in materialMap) {
      object.material = materialMap[materialName as keyof typeof materialMap];
    } else {
      console.log(materialName);
      object.material = defaultBlueMaterial;
    }
  });

  if (options.signal.aborted) {
    return;
  }
  await runPathTracer("render-target", model, {
    targetSamples: 100,
    ...options,
  });
}

function usePathTracer(
  modelUrl: string,
  materialMap: Record<string, OpenPBRMaterial>,
  options: Parameters<typeof runPathTracer>[2],
) {
  const [canvasSize, setCanvasSize] = React.useState<number | null>(null);

  React.useEffect(() => {
    const destroyController = new AbortController();
    const signal = destroyController.signal;

    if (canvasSize) {
      init(modelUrl, materialMap, {
        signal,
        kTextureWidth: canvasSize,
        ...options,
      });
    }

    return () => {
      destroyController.abort();
    };
  }, [canvasSize, options]);

  React.useEffect(() => {
    setCanvasSize(window.innerWidth > 512 ? 512 : 368);
  }, []);

  return canvasSize === null ? null : (
    <canvas width={canvasSize} height={canvasSize} id="render-target"></canvas>
  );
}

export default usePathTracer;

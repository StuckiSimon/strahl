import React from "react";
import { OpenPBRMaterial, runPathTracer } from "strahl";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

let modelCacheMap = new Map<string, any>();

const gltfLoader = new GLTFLoader();

async function loadGltf(url: string) {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

const defaultBlueMaterial = new OpenPBRMaterial();
defaultBlueMaterial.oBaseColor = [0.0, 0.9, 1.0];

async function init(
  target: string,
  modelUrl: string,
  materialMap: Record<string, OpenPBRMaterial>,
  options: Parameters<typeof runPathTracer>[2],
) {
  let modelCache = modelCacheMap.get(modelUrl);
  if (modelCache === undefined) {
    modelCache = await loadGltf(modelUrl);
    modelCacheMap.set(modelUrl, modelCache);
  }
  const model = {
    scene: modelCache.scene.clone(),
  };

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
  await runPathTracer(target, model, {
    targetSamples: 100,
    ...options,
  });
}

function usePathTracer(
  modelUrl: string,
  materialMap: Record<string, OpenPBRMaterial>,
  options: Parameters<typeof runPathTracer>[2],
) {
  const id = React.useId();
  const [canvasSize, setCanvasSize] = React.useState<number | null>(null);

  React.useEffect(() => {
    const destroyController = new AbortController();
    const signal = destroyController.signal;

    if (canvasSize) {
      init(`render-target-${id}`, modelUrl, materialMap, {
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
    <canvas
      width={canvasSize}
      height={canvasSize}
      id={`render-target-${id}`}
    ></canvas>
  );
}

export default usePathTracer;

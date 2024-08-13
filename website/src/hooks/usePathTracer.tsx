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

async function init(
  modelUrl: string,
  materialMap: Record<string, OpenPBRMaterial>,
  signal: AbortSignal,
  kSize: number,
) {
  let model = modelCache.get(modelUrl);
  if (model === undefined) {
    model = await loadGltf(modelUrl);
    modelCache.set(modelUrl, model);
  }

  const defaultBlueMaterial = new OpenPBRMaterial();
  defaultBlueMaterial.oBaseColor = [0.0, 0.9, 1.0];

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

  if (signal.aborted) {
    return;
  }
  await runPathTracer("render-target", model, {
    targetSamples: 100,
    signal,
    kTextureWidth: kSize,
  });
}

function usePathTracer(
  modelUrl: string,
  materialMap: Record<string, OpenPBRMaterial>,
) {
  const [canvasSize, setCanvasSize] = React.useState<number | null>(null);
  const loadingState = React.useRef(false);

  React.useEffect(() => {
    const destroyController = new AbortController();
    const signal = destroyController.signal;

    if (!loadingState.current && canvasSize) {
      loadingState.current = true;
      init(modelUrl, materialMap, signal, canvasSize);
    }

    return () => {
      destroyController.abort();
    };
  }, [canvasSize]);

  React.useEffect(() => {
    setCanvasSize(window.innerWidth > 512 ? 512 : 368);
  }, []);

  return canvasSize === null ? null : (
    <canvas width={canvasSize} height={canvasSize} id="render-target"></canvas>
  );
}

export default usePathTracer;

import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { Group } from "three";

const gltfLoader = new GLTFLoader();

export async function loadGltf(url: string): Promise<{ scene: Group }> {
  return new Promise((resolve, reject) => {
    gltfLoader.load(url, resolve, undefined, reject);
  });
}

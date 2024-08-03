import tracerShader from "./tracer-shader.wgsl?raw";

type Params = {
  imageWidth: number;
  imageHeight: number;
  maxWorkgroupDimension: number;
  maxBvhStackDepth: number;
};

const PARAM_PLACEHOLDER_MAP: Record<keyof Params, string> = {
  imageWidth: "imageWidth",
  imageHeight: "imageHeight",
  maxWorkgroupDimension: "maxWorkgroupDimension",
  maxBvhStackDepth: "maxBvhStackDepth",
};

export default function build(params: Params) {
  const placeholders = Object.entries(PARAM_PLACEHOLDER_MAP) as [
    keyof Params,
    string,
  ][];
  return placeholders.reduce((aggregate, [key, value]) => {
    return aggregate.replaceAll(`\${${value}}`, `${params[key]}`);
  }, tracerShader);
}

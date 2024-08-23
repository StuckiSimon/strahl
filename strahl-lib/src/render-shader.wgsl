const pos = array(
  // 1
  vec2f(-1.0, -1.0),
  vec2f(1.0, -1.0),
  vec2f(-1.0, 1.0),
  // 2
  vec2f( -1.0,  1.0),
  vec2f( 1.0,  -1.0),
  vec2f( 1.0,  1.0),
);

struct VertexInput {
  @builtin(vertex_index) instance: u32,
};

struct VertexOutput {
  @builtin(position) pos: vec4f,
  @location(0) texcoord: vec2f,
};

fn convertToZeroOne(position: vec2f) -> vec2f {
  return (position + 1.0) / 2;
}

fn sample(uv: vec2<f32>) -> vec4<f32> {
  return textureSample(texture, texture_sampler, uv);
}

const INV_SQRT_OF_2PI = 0.39894228040143267793994605993439;
const INV_PI = 0.31830988618379067153776752674503;

// CODE#DENOISE-FILTERING
// Based on https://github.com/BrutPitt/glslSmartDeNoise
fn denoise(uv: vec2f) -> vec4f {
  let sigma = uniformData.denoiseSigma;
  let kSigma = uniformData.denoiseKSigma;
  let threshold = uniformData.denoiseThreshold;

  let radius = round(kSigma * sigma);
  let radQ = radius * radius;

  let invSigmaQx2 = .5 / (sigma * sigma);
  let invSigmaQx2PI = INV_PI * invSigmaQx2;

  let invThresholdSqx2 = .5 / (threshold * threshold);
  let invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold;

  let centrPx = sample(uv);

  let isClearColorActive = centrPx.a == 0.0;

  var zBuff = 0.0;
  var aBuff = vec4f(0);
  let size = vec2f(f32(uniformData.textureWidth), f32(uniformData.textureHeight));

  var d: vec2f;
  for (d.x=-radius; d.x <= radius; d.x += 1.0) {
    let pt = sqrt(radQ - d.x * d.x);
    for (d.y=-pt; d.y <= pt; d.y += 1.0) {
      let blurFactor = exp(-dot(d, d) * invSigmaQx2) * invSigmaQx2PI;

      let walkPx = sample(uv + d / size);
      let dC = walkPx - centrPx;
      let deltaFactor = exp(-dot(dC, dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;

      zBuff += deltaFactor;
      aBuff += deltaFactor * walkPx;
    }
  }

  // sampling is only permitted in uniform control flow, therefore we do this trickery
  if (isClearColorActive) {
    return centrPx;
  }
  return aBuff / zBuff;
}

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let xy = pos[input.instance];
  output.pos = vec4f(xy, 0, 1);

  let baseCoord = convertToZeroOne(vec2f(xy.x, -xy.y));
  output.texcoord = vec2f(baseCoord.x, baseCoord.y);
  return output;
}

struct UniformData {
  enableDenoise: i32,
  textureWidth: i32,
  textureHeight: i32,
  denoiseSigma: f32,
  denoiseKSigma: f32,
  denoiseThreshold: f32,
}

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;
@group(0) @binding(2) var<uniform> uniformData: UniformData;

// CODE#TONE-MAPPER
// Khronos PBR neutral tone mapper
// See: https://github.com/KhronosGroup/ToneMapping/blob/main/PBR_Neutral/pbrNeutral.glsl
fn khronosPBRNeutralToneMapping(colorP: vec3f) -> vec3f {
  var color = colorP;
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;

  let x = min(color.r, min(color.g, color.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  color -= offset;

  let peak = max(color.r, max(color.g, color.b));
  if (peak < startCompression) {
    return color;
  }

  let d = 1.0 - startCompression;
  let newPeak = 1. - d * d / (peak + d - startCompression);
  color *= newPeak / peak;

  let g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
  return mix(color, newPeak * vec3(1, 1, 1), g);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  var color: vec4f;
  if (uniformData.enableDenoise == 1) {
    color = denoise(input.texcoord);
  } else {
    color = sample(input.texcoord);
  }

  // respect clear color
  if (color.a == 0.0) {
    return vec4f(color.rgb, 0.0);
  }

  let tonemappedColor = khronosPBRNeutralToneMapping(color.rgb);
  return vec4f(tonemappedColor, 1.0);
}

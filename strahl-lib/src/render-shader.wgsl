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

@vertex
fn vertexMain(input: VertexInput) -> VertexOutput {
  var output: VertexOutput;
  let xy = pos[input.instance];
  output.pos = vec4f(xy, 0, 1);

  let baseCoord = convertToZeroOne(vec2f(xy.x, -xy.y));
  output.texcoord = vec2f(baseCoord.x, 1.0 - baseCoord.y);
  return output;
}

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;

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
  let hdrColor = textureSample(texture, texture_sampler, input.texcoord);

  // respect clear color
  if (hdrColor.a == 0.0) {
    return vec4f(hdrColor.rgb, 0.0);
  }

  let tonemappedColor = khronosPBRNeutralToneMapping(hdrColor.rgb);
  return vec4f(tonemappedColor, 1.0);
}

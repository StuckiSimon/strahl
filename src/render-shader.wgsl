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

  let baseCoord = convertToZeroOne(vec2f(xy.x, xy.y));
  output.texcoord = vec2f(baseCoord.x, 1.0 - baseCoord.y);
  return output;
}

@group(0) @binding(0) var texture_sampler: sampler;
@group(0) @binding(1) var texture: texture_2d<f32>;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  return textureSample(texture, texture_sampler, input.texcoord);
}
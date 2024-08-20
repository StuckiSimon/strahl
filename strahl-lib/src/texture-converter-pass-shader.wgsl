@group(0) @binding(0) var readTexture: texture_storage_2d<rgba32float, read>;
// float32 rgba
@group(0) @binding(1) var<storage, read_write> targetBuffer: array<vec4f>;

@compute
@workgroup_size(${maxWorkgroupDimension}, ${maxWorkgroupDimension}, 1)
fn computeMain(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let position = vec2<i32>(i32(globalId.x), i32(globalId.y));
  let previousColor = textureLoad(readTexture, position);
  targetBuffer[position.x + position.y * ${imageWidth}] = previousColor;
}

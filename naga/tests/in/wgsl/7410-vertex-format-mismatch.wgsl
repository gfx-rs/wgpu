// Regression test for https://github.com/gfx-rs/wgpu/issues/7410
//
// The vertex buffer description (which comes from GPURenderPipelineDescriptor in the
// original) defines `texcoord` with dimension 3, but the shader consumes it with
// dimension 2.
//
// As of 2025-03-24, the committed msl output snapshot is invalid because it attempts
// to cast a `metal::float3` into a `metal::float2`.

struct Out {
  @builtin(position)
  position: vec4f,
  @location(0)
  color: vec4f,
  @location(1)
  texcoord: vec2f,
}

@vertex
fn main(@location(0) position: vec4f, @location(1) color: vec4f, @location(2) texcoord: vec2f) -> Out {
  return Out(position, color, texcoord);
}

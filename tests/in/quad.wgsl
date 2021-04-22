// vertex
let c_scale: f32 = 1.2;

struct VertexOutput {
  [[location(0)]] uv : vec2<f32>;
  [[builtin(position)]] position : vec4<f32>;
};

[[stage(vertex)]]
fn main(
  [[location(0)]] pos : vec2<f32>,
  [[location(1)]] uv : vec2<f32>,
) -> VertexOutput {
  return VertexOutput(uv, vec4<f32>(c_scale * pos, 0.0, 1.0));
}

// fragment
[[group(0), binding(0)]] var u_texture : texture_2d<f32>;
[[group(0), binding(1)]] var u_sampler : sampler;

[[stage(fragment)]]
fn main([[location(0)]] uv : vec2<f32>) -> [[location(0)]] vec4<f32> {
  let color = textureSample(u_texture, u_sampler, uv);
  if (color.a == 0.0) {
    discard;
  }
  // forcing the expression here to be emitted in order to check the
  // uniformity of the control flow a bit more strongly.
  let premultiplied = color.a * color;
  return premultiplied;
}

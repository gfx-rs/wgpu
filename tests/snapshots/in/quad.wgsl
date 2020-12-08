# vertex
const c_scale: f32 = 1.2;
[[location(0)]] var<in> a_pos : vec2<f32>;
[[location(1)]] var<in> a_uv : vec2<f32>;
[[location(0)]] var<out> v_uv : vec2<f32>;
[[builtin(position)]] var<out> o_position : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
  v_uv = a_uv;
  o_position = vec4<f32>(c_scale * a_pos, 0.0, 1.0);
  return;
}

# fragment
[[location(0)]] var<in> v_uv : vec2<f32>;
[[group(0), binding(0)]] var u_texture : texture_sampled_2d<f32>;
[[group(0), binding(1)]] var u_sampler : sampler;
[[location(0)]] var<out> o_color : vec4<f32>;

[[stage(fragment)]]
fn main() -> void {
  o_color = textureSample(u_texture, u_sampler, v_uv);
  return;
}

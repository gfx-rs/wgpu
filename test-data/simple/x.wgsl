# vertex
[[location(0)]] var<in> a_pos : vec2<f32>;
[[location(0)]] var<out> o_pos : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
  var w: f32 = 1.0;
  o_pos = vec4<f32>(a_pos, 0.0, w);
  return;
}

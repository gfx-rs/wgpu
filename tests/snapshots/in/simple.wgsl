# vertex
[[builtin(position)]] var<out> o_position : vec4<f32>;

[[stage(vertex)]]
fn main() -> void {
  o_position = vec4<f32>(1);
  return;
}

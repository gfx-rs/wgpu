# vertex
[[builtin position]] var<out> o_position : vec4<f32>;

fn main() -> void {
  o_position = vec4<f32>(1);
  return;
}
entry_point vertex as "main" = main;

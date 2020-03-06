# vertex
const c_scale: f32 = 1.2;
[[location 0]] var<in> a_pos : vec2<f32>;
[[location 1]] var<in> a_uv : vec2<f32>;
[[location 0]] var<out> v_uv : vec2<f32>;
[[builtin position]] var<out> o_position : vec4<f32>;

fn main_vert() -> void {
  o_position = vec4<f32>(c_scale * a_pos, 0.0, 1.0);
  return;
}
entry_point vertex as "main" = main_vert;

# fragment
[[location 0]] var<in> a_uv : vec2<f32>;
#layout(set = 0, binding = 0) uniform texture2D u_texture;
#layout(set = 0, binding = 1) uniform sampler u_sampler;
[[location 0]] var<out> o_color : vec4<f32>;

fn main_frag() -> void {
  o_color = vec4<f32>(1, 0, 0, 1); #TODO: sample
  return;
}
entry_point fragment as "main" = main_frag;

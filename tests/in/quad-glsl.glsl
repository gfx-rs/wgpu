#version 450
// vertex
const float c_scale = 1.2;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;

void vert_main() {
  v_uv = a_uv;
  gl_Position = vec4(c_scale * a_pos, 0.0, 1.0);
}

// fragment
layout(location = 0) in vec2 v_uv;
#ifdef TEXTURE
layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;
#endif
layout(location = 0) out vec4 o_color;

void frag_main() {
#ifdef TEXTURE
  o_color = texture(sampler2D(u_texture, u_sampler), v_uv);
#else
  o_color = vec4(1.0, 1.0, 1.0, 1.0);
#endif
}

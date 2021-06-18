#version 450
layout(location = 0) in vec2 v_uv;
#ifdef TEXTURE
layout(set = 0, binding = 0) uniform texture2D u_texture;
layout(set = 0, binding = 1) uniform sampler u_sampler;
#endif
layout(location = 0) out vec4 o_color;

void main() {
#ifdef TEXTURE
  o_color = texture(sampler2D(u_texture, u_sampler), v_uv);
#else
  o_color = vec4(1.0, 1.0, 1.0, 1.0);
#endif
}

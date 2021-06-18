#version 450
const float c_scale = 1.2;

layout(location = 0) in vec2 a_pos;
layout(location = 1) in vec2 a_uv;
layout(location = 0) out vec2 v_uv;

void main() {
  v_uv = a_uv;
  gl_Position = vec4(c_scale * a_pos, 0.0, 1.0);
}

#version 450
layout(location = 0) in vec2 a_pos;

void main() {
  gl_Position = vec4(clamp(a_pos, 0.0, 1.0), 0.0, 1.0);
}

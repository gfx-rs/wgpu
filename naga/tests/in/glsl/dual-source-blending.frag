#version 450

layout(location = 0, index = 0) out vec4 output0;
layout(location = 0, index = 1) out vec4 output1;

void main() {
  output0 = vec4(1.0, 0.0, 1.0, 0.0);
  output1 = vec4(0.0, 1.0, 0.0, 1.0);
}
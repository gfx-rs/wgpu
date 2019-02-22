#version 450

layout(location = 0) out vec4 o_Target;

layout(set = 0, binding = 2) uniform texture2D u_ShadowTexture;
layout(set = 0, binding = 3) uniform sampler u_ShadowSampler;

void main() {
	o_Target = vec4(1.0);
}

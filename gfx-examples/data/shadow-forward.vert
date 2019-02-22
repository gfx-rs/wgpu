#version 450

layout(location = 0) in ivec4 a_Pos;
layout(location = 1) in ivec4 a_Normal;

struct Light {
	mat4 proj;
	vec4 color;
};

layout(set = 0, binding = 0) uniform Globals {
    mat4 u_ViewProj;
    uvec4 u_NumLights;
};
layout(set = 0, binding = 1) uniform Lights {
    Light u_Lights[];
};
layout(set = 1, binding = 0) uniform Entity {
    mat4 u_World;
    vec4 u_Color;
};

void main() {
    gl_Position = u_ViewProj * u_World * vec4(a_Pos);
}

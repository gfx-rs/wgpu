#version 310 es

precision highp float;
precision highp int;

struct Globals {
    mat4x4 view_proj;
    uvec4 num_lights;
};
struct Entity {
    mat4x4 world;
    vec4 color;
};
struct VertexOutput {
    vec4 proj_position;
    vec3 world_normal;
    vec4 world_position;
};
struct Light {
    mat4x4 proj;
    vec4 pos;
    vec4 color;
};
uniform Globals_block_0Vertex { Globals _group_0_binding_0_vs; };

uniform Entity_block_1Vertex { Entity _group_1_binding_0_vs; };

layout(location = 0) in ivec4 _p2vs_location0;

void main() {
    ivec4 position = _p2vs_location0;
    mat4x4 _e4 = _group_0_binding_0_vs.view_proj;
    mat4x4 _e6 = _group_1_binding_0_vs.world;
    gl_Position = ((_e4 * _e6) * vec4(position));
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}


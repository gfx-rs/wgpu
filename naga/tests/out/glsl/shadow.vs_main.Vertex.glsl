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
const vec3 c_ambient = vec3(0.05, 0.05, 0.05);
const uint c_max_lights = 10u;

uniform Globals_block_0Vertex { Globals _group_0_binding_0_vs; };

uniform Entity_block_1Vertex { Entity _group_1_binding_0_vs; };

layout(location = 0) in ivec4 _p2vs_location0;
layout(location = 1) in ivec4 _p2vs_location1;
layout(location = 0) smooth out vec3 _vs2fs_location0;
layout(location = 1) smooth out vec4 _vs2fs_location1;

void main() {
    ivec4 position = _p2vs_location0;
    ivec4 normal = _p2vs_location1;
    VertexOutput out_ = VertexOutput(vec4(0.0), vec3(0.0), vec4(0.0));
    mat4x4 w = _group_1_binding_0_vs.world;
    mat4x4 _e7 = _group_1_binding_0_vs.world;
    vec4 world_pos = (_e7 * vec4(position));
    out_.world_normal = (mat3x3(w[0].xyz, w[1].xyz, w[2].xyz) * vec3(normal.xyz));
    out_.world_position = world_pos;
    mat4x4 _e26 = _group_0_binding_0_vs.view_proj;
    out_.proj_position = (_e26 * world_pos);
    VertexOutput _e28 = out_;
    gl_Position = _e28.proj_position;
    _vs2fs_location0 = _e28.world_normal;
    _vs2fs_location1 = _e28.world_position;
    gl_Position.yz = vec2(-gl_Position.y, gl_Position.z * 2.0 - gl_Position.w);
    return;
}


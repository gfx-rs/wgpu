///////////////////////////////////
// Entry point: "vs_main" (vert) //
///////////////////////////////////
#version 460

struct Globals
{
    mat4 view_proj;
    uvec4 num_lights;
};

struct Entity
{
    mat4 world;
    vec4 color;
};

struct VertexOutput
{
    vec4 proj_position;
    vec3 world_normal;
    vec4 world_position;
};

struct Light
{
    mat4 proj;
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 0, std140) uniform u_globals
{
    Globals _m0;
} u_globals_1;

layout(set = 1, binding = 0, std140) uniform u_entity
{
    Entity _m0;
} u_entity_1;

layout(set = 0, binding = 1, std430) readonly buffer s_lights
{
    Light _m0[];
} s_lights_1;

layout(set = 0, binding = 1, std140) uniform u_lights
{
    Light _m0[10];
} u_lights_1;

layout(set = 0, binding = 2) uniform texture2DArray t_shadow;
layout(set = 0, binding = 3) uniform sampler sampler_shadow;

layout(location = 0) in ivec4 position;
layout(location = 1) in ivec4 normal;
layout(location = 0) out vec3 world_normal;
layout(location = 1) out vec4 world_position;

void main()
{
    VertexOutput _out = VertexOutput(vec4(0.0), vec3(0.0), vec4(0.0));
    vec4 _102 = u_entity_1._m0.world * vec4(position);
    _out.world_normal = mat3(u_entity_1._m0.world[0].xyz, u_entity_1._m0.world[1].xyz, u_entity_1._m0.world[2].xyz) * vec3(normal.xyz);
    _out.world_position = _102;
    _out.proj_position = u_globals_1._m0.view_proj * _102;
    gl_Position = _out.proj_position;
    gl_Position.y = -gl_Position.y;
    world_normal = _out.world_normal;
    world_position = _out.world_position;
}


///////////////////////////////////
// Entry point: "fs_main" (frag) //
///////////////////////////////////
#version 460

struct Globals
{
    mat4 view_proj;
    uvec4 num_lights;
};

struct Entity
{
    mat4 world;
    vec4 color;
};

struct VertexOutput
{
    vec4 proj_position;
    vec3 world_normal;
    vec4 world_position;
};

struct Light
{
    mat4 proj;
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 0, std140) uniform u_globals
{
    Globals _m0;
} u_globals_1;

layout(set = 1, binding = 0, std140) uniform u_entity
{
    Entity _m0;
} u_entity_1;

layout(set = 0, binding = 1, std430) readonly buffer s_lights
{
    Light _m0[];
} s_lights_1;

layout(set = 0, binding = 1, std140) uniform u_lights
{
    Light _m0[10];
} u_lights_1;

layout(set = 0, binding = 2) uniform texture2DArray t_shadow;
layout(set = 0, binding = 3) uniform samplerShadow sampler_shadow;

layout(location = 0) in vec3 world_normal;
layout(location = 1) in vec4 world_position;
layout(location = 0) out vec4 _142;

float fetch_shadow(uint light_id, vec4 homogeneous_coords)
{
    if (homogeneous_coords.w <= 0.0)
    {
        return 1.0;
    }
    float _61 = 1.0 / homogeneous_coords.w;
    return textureGrad(sampler2DArrayShadow(t_shadow, sampler_shadow), vec4(vec3(((homogeneous_coords.xy * vec2(0.5, -0.5)) * _61) + vec2(0.5), float(int(light_id))), homogeneous_coords.z * _61), vec2(0.0), vec2(0.0));
}

void main()
{
    vec3 color = vec3(0.0500000007450580596923828125);
    uint i = 0u;
    uvec2 loop_bound = uvec2(4294967295u);
    VertexOutput _133 = VertexOutput(gl_FragCoord, world_normal, world_position);
    vec3 _155 = normalize(_133.world_normal);
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (!(i < min(u_globals_1._m0.num_lights.x, 10u)))
        {
            break;
        }
        color += (s_lights_1._m0[i].color.xyz * (fetch_shadow(i, s_lights_1._m0[i].proj * _133.world_position) * max(0.0, dot(_155, normalize(s_lights_1._m0[i].pos.xyz - _133.world_position.xyz)))));
        i++;
        continue;
    }
    _142 = vec4(color, 1.0) * u_entity_1._m0.color;
}


///////////////////////////////////////////////////
// Entry point: "fs_main_without_storage" (frag) //
///////////////////////////////////////////////////
#version 460

struct Globals
{
    mat4 view_proj;
    uvec4 num_lights;
};

struct Entity
{
    mat4 world;
    vec4 color;
};

struct VertexOutput
{
    vec4 proj_position;
    vec3 world_normal;
    vec4 world_position;
};

struct Light
{
    mat4 proj;
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 0, std140) uniform u_globals
{
    Globals _m0;
} u_globals_1;

layout(set = 1, binding = 0, std140) uniform u_entity
{
    Entity _m0;
} u_entity_1;

layout(set = 0, binding = 1, std430) readonly buffer s_lights
{
    Light _m0[];
} s_lights_1;

layout(set = 0, binding = 1, std140) uniform u_lights
{
    Light _m0[10];
} u_lights_1;

layout(set = 0, binding = 2) uniform texture2DArray t_shadow;
layout(set = 0, binding = 3) uniform samplerShadow sampler_shadow;

layout(location = 0) in vec3 world_normal;
layout(location = 1) in vec4 world_position;
layout(location = 0) out vec4 _227;

float fetch_shadow(uint light_id, vec4 homogeneous_coords)
{
    if (homogeneous_coords.w <= 0.0)
    {
        return 1.0;
    }
    float _61 = 1.0 / homogeneous_coords.w;
    return textureGrad(sampler2DArrayShadow(t_shadow, sampler_shadow), vec4(vec3(((homogeneous_coords.xy * vec2(0.5, -0.5)) * _61) + vec2(0.5), float(int(light_id))), homogeneous_coords.z * _61), vec2(0.0), vec2(0.0));
}

void main()
{
    vec3 color = vec3(0.0500000007450580596923828125);
    uint i = 0u;
    uvec2 loop_bound = uvec2(4294967295u);
    VertexOutput _220 = VertexOutput(gl_FragCoord, world_normal, world_position);
    vec3 _239 = normalize(_220.world_normal);
    for (;;)
    {
        if (all(equal(uvec2(0u), loop_bound)))
        {
            break;
        }
        loop_bound -= uvec2(uint(loop_bound.y == 0u), 1u);
        if (!(i < min(u_globals_1._m0.num_lights.x, 10u)))
        {
            break;
        }
        color += (u_lights_1._m0[i].color.xyz * (fetch_shadow(i, u_lights_1._m0[i].proj * _220.world_position) * max(0.0, dot(_239, normalize(u_lights_1._m0[i].pos.xyz - _220.world_position.xyz)))));
        i++;
        continue;
    }
    _227 = vec4(color, 1.0) * u_entity_1._m0.color;
}


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

uniform Globals_block_0Fragment { Globals _group_0_binding_0_fs; };

uniform Entity_block_1Fragment { Entity _group_1_binding_0_fs; };

uniform type_7_block_2Fragment { Light _group_0_binding_1_fs[10]; };

uniform highp sampler2DArrayShadow _group_0_binding_2_fs;

layout(location = 0) smooth in vec3 _vs2fs_location0;
layout(location = 1) smooth in vec4 _vs2fs_location1;
layout(location = 0) out vec4 _fs2p_location0;

float fetch_shadow(uint light_id, vec4 homogeneous_coords) {
    if ((homogeneous_coords.w <= 0.0)) {
        return 1.0;
    }
    vec2 flip_correction = vec2(0.5, -0.5);
    float proj_correction = (1.0 / homogeneous_coords.w);
    vec2 light_local = (((homogeneous_coords.xy * flip_correction) * proj_correction) + vec2(0.5, 0.5));
    float _e24 = textureGrad(_group_0_binding_2_fs, vec4(light_local, int(light_id), (homogeneous_coords.z * proj_correction)), vec2(0.0), vec2(0.0));
    return _e24;
}

void main() {
    VertexOutput in_1 = VertexOutput(gl_FragCoord, _vs2fs_location0, _vs2fs_location1);
    vec3 color_1 = vec3(0.0);
    uint i_1 = 0u;
    vec3 normal_1 = normalize(in_1.world_normal);
    color_1 = c_ambient;
    i_1 = 0u;
    bool loop_init = true;
    while(true) {
        if (!loop_init) {
            uint _e40 = i_1;
            i_1 = (_e40 + 1u);
        }
        loop_init = false;
        uint _e7 = i_1;
        uint _e11 = _group_0_binding_0_fs.num_lights.x;
        if ((_e7 < min(_e11, c_max_lights))) {
        } else {
            break;
        }
        {
            uint _e16 = i_1;
            Light light = _group_0_binding_1_fs[_e16];
            uint _e19 = i_1;
            float _e23 = fetch_shadow(_e19, (light.proj * in_1.world_position));
            vec3 light_dir = normalize((light.pos.xyz - in_1.world_position.xyz));
            float diffuse = max(0.0, dot(normal_1, light_dir));
            vec3 _e37 = color_1;
            color_1 = (_e37 + ((_e23 * diffuse) * light.color.xyz));
        }
    }
    vec3 _e42 = color_1;
    vec4 _e47 = _group_1_binding_0_fs.color;
    _fs2p_location0 = (vec4(_e42, 1.0) * _e47);
    return;
}


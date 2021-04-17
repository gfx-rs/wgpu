#version 310 es

precision highp float;

struct Light {
    mat4x4 proj;
    vec4 pos;
    vec4 color;
};

uniform Globals_block_0 {
    uvec4 num_lights;
} _group_0_binding_0;

readonly buffer Lights_block_1 {
    Light data[];
} _group_0_binding_1;

uniform highp sampler2DArrayShadow _group_0_binding_2;

smooth layout(location = 0) in vec3 _vs2fs_location0;
smooth layout(location = 1) in vec4 _vs2fs_location1;
layout(location = 0) out vec4 _fs2p_location0;

float fetch_shadow(uint light_id, vec4 homogeneous_coords) {
    if((homogeneous_coords[3] <= 0.0)) {
        return 1.0;
    }
    float _expr28 = textureGrad(_group_0_binding_2, vec4((((vec2(homogeneous_coords[0], homogeneous_coords[1]) * vec2(0.5, -0.5)) / vec2(homogeneous_coords[3])) + vec2(0.5, 0.5)), int(light_id), (homogeneous_coords[2] / homogeneous_coords[3])), vec2(0, 0), vec2(0,0));
    return _expr28;
}

void main() {
    vec3 raw_normal = _vs2fs_location0;
    vec4 position = _vs2fs_location1;
    vec3 color1 = vec3(0.05, 0.05, 0.05);
    uint i = 0u;
    while(true) {
        if((i >= min(_group_0_binding_0.num_lights[0], 10u))) {
            break;
        }
        Light _expr21 = _group_0_binding_1.data[i];
        float _expr25 = fetch_shadow(i, (_expr21.proj * position));
        color1 = (color1 + ((_expr25 * max(0.0, dot(normalize(raw_normal), normalize((vec3(_expr21.pos[0], _expr21.pos[1], _expr21.pos[2]) - vec3(position[0], position[1], position[2])))))) * vec3(_expr21.color[0], _expr21.color[1], _expr21.color[2])));
        i = (i + 1u);
    }
    _fs2p_location0 = vec4(color1, 1.0);
    return;
}


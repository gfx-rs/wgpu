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

smooth in vec3 _vs2fs_location0;
smooth in vec4 _vs2fs_location1;
layout(location = 0) out vec4 _fs2p_location0;

float fetch_shadow(uint light_id, vec4 homogeneous_coords) {
    if((homogeneous_coords.w <= 0.0)) {
        return 1.0;
    }
    vec2 flip_correction = vec2(0.5, -0.5);
    vec2 light_local = (((homogeneous_coords.xy * flip_correction) / vec2(homogeneous_coords.w)) + vec2(0.5, 0.5));
    float _expr26 = textureGrad(_group_0_binding_2, vec4(light_local, int(light_id), (homogeneous_coords.z / homogeneous_coords.w)), vec2(0, 0), vec2(0,0));
    return _expr26;
}

void main() {
    vec3 raw_normal = _vs2fs_location0;
    vec4 position = _vs2fs_location1;
    vec3 color1 = vec3(0.05, 0.05, 0.05);
    uint i = 0u;
    vec3 normal = normalize(raw_normal);
    while(true) {
        uint _expr12 = i;
        uvec4 _expr14 = _group_0_binding_0.num_lights;
        if((_expr12 >= min(_expr14.x, 10u))) {
            break;
        }
        uint _expr19 = i;
        Light light = _group_0_binding_1.data[_expr19];
        uint _expr22 = i;
        float _expr25 = fetch_shadow(_expr22, (light.proj * position));
        vec3 light_dir = normalize((light.pos.xyz - position.xyz));
        float diffuse = max(0.0, dot(normal, light_dir));
        vec3 _expr34 = color1;
        color1 = (_expr34 + ((_expr25 * diffuse) * light.color.xyz));
        uint _expr40 = i;
        i = (_expr40 + 1u);
    }
    vec3 _expr43 = color1;
    _fs2p_location0 = vec4(_expr43, 1.0);
    return;
}


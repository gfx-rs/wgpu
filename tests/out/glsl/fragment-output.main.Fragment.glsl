#version 310 es

precision highp float;
precision highp int;

struct FragmentOutput {
    vec4 vec4f;
    ivec4 vec4i;
    uvec4 vec4u;
    vec3 vec3f;
    ivec3 vec3i;
    uvec3 vec3u;
    vec2 vec2f;
    ivec2 vec2i;
    uvec2 vec2u;
    float scalarf;
    int scalari;
    uint scalaru;
};
layout(location = 0) out vec4 _fs2p_location0;
layout(location = 1) out ivec4 _fs2p_location1;
layout(location = 2) out uvec4 _fs2p_location2;
layout(location = 3) out vec3 _fs2p_location3;
layout(location = 4) out ivec3 _fs2p_location4;
layout(location = 5) out uvec3 _fs2p_location5;
layout(location = 6) out vec2 _fs2p_location6;
layout(location = 7) out ivec2 _fs2p_location7;
layout(location = 8) out uvec2 _fs2p_location8;
layout(location = 9) out float _fs2p_location9;
layout(location = 10) out int _fs2p_location10;
layout(location = 11) out uint _fs2p_location11;

void main() {
    FragmentOutput output_ = FragmentOutput(vec4(0.0), ivec4(0), uvec4(0u), vec3(0.0), ivec3(0), uvec3(0u), vec2(0.0), ivec2(0), uvec2(0u), 0.0, 0, 0u);
    output_.vec4f = vec4(0.0);
    output_.vec4i = ivec4(0);
    output_.vec4u = uvec4(0u);
    output_.vec3f = vec3(0.0);
    output_.vec3i = ivec3(0);
    output_.vec3u = uvec3(0u);
    output_.vec2f = vec2(0.0);
    output_.vec2i = ivec2(0);
    output_.vec2u = uvec2(0u);
    output_.scalarf = 0.0;
    output_.scalari = 0;
    output_.scalaru = 0u;
    FragmentOutput _e34 = output_;
    _fs2p_location0 = _e34.vec4f;
    _fs2p_location1 = _e34.vec4i;
    _fs2p_location2 = _e34.vec4u;
    _fs2p_location3 = _e34.vec3f;
    _fs2p_location4 = _e34.vec3i;
    _fs2p_location5 = _e34.vec3u;
    _fs2p_location6 = _e34.vec2f;
    _fs2p_location7 = _e34.vec2i;
    _fs2p_location8 = _e34.vec2u;
    _fs2p_location9 = _e34.scalarf;
    _fs2p_location10 = _e34.scalari;
    _fs2p_location11 = _e34.scalaru;
    return;
}


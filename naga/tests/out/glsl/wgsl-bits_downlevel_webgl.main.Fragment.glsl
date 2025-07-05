#version 300 es

precision highp float;
precision highp int;


void main() {
    int i = 0;
    ivec2 i2_ = ivec2(0);
    ivec3 i3_ = ivec3(0);
    ivec4 i4_ = ivec4(0);
    uint u = 0u;
    uvec2 u2_ = uvec2(0u);
    uvec3 u3_ = uvec3(0u);
    uvec4 u4_ = uvec4(0u);
    vec2 f2_ = vec2(0.0);
    vec4 f4_ = vec4(0.0);
    ivec4 _e23 = i4_;
    u = uint((_e23[0] & 0xFF) | ((_e23[1] & 0xFF) << 8) | ((_e23[2] & 0xFF) << 16) | ((_e23[3] & 0xFF) << 24));
    uvec4 _e25 = u4_;
    u = (_e25[0] & 0xFFu) | ((_e25[1] & 0xFFu) << 8) | ((_e25[2] & 0xFFu) << 16) | ((_e25[3] & 0xFFu) << 24);
    uint _e27 = u;
    f4_ = (vec4(ivec4(_e27 << 24, _e27 << 16, _e27 << 8, _e27) >> 24) / 127.0);
    uint _e29 = u;
    f4_ = (vec4(_e29 & 0xFFu, _e29 >> 8 & 0xFFu, _e29 >> 16 & 0xFFu, _e29 >> 24) / 255.0);
    uint _e31 = u;
    f2_ = unpackSnorm2x16(_e31);
    uint _e33 = u;
    f2_ = unpackUnorm2x16(_e33);
    return;
}


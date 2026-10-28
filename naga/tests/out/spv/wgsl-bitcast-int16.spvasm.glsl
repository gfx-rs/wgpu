#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    vec2 _54 = vec2(0.0);
    float _48 = 0.0;
    u16vec4 _42 = u16vec4(1us, 2us, 3us, 4us);
    i16vec2 _36 = i16vec2(1s, 2s);
    ivec2 _50 = ivec2(0);
    int _44 = 0;
    u16vec2 _38 = u16vec2(1us, 2us);
    uvec2 _52 = uvec2(0u);
    uint _46 = 0u;
    i16vec4 _40 = i16vec4(1s, 2s, 3s, 4s);
    _44 = packInt2x16(_36);
    _46 = packUint2x16(_38);
    _48 = (_38);
    _52 = (_40);
    _54 = (_42);
    _36 = (_46);
    _38 = (_48);
    _40 = (_54);
    _42 = (_50);
}


#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _14
{
    uint _m0;
    int _m1;
    float _m2;
    uint64_t _m3;
    u64vec2 _m4;
    u64vec3 _m5;
    u64vec4 _m6;
    int64_t _m7;
    i64vec2 _m8;
    i64vec3 _m9;
    i64vec4 _m10;
    uint64_t _m11;
};

struct _18
{
    uint64_t _m0[2];
    int64_t _m1[2];
};

layout(set = 0, binding = 0, std140) uniform _24_23
{
    _14 _m0;
} _23;

layout(set = 0, binding = 1, std430) readonly buffer _27_26
{
    _14 _m0;
} _26;

layout(set = 0, binding = 2, std430) readonly buffer _30_29
{
    _18 _m0;
} _29;

layout(set = 0, binding = 3, std430) buffer _33_32
{
    _14 _m0;
} _32;

layout(set = 0, binding = 4, std430) buffer _36_35
{
    _18 _m0;
} _35;

int64_t _21 = 1l;

uint64_t _222(uint64_t _221)
{
    uint64_t _232 = 20ul;
    _232 += ((31ul + 18446744073709551615ul) - 18446744073709551615ul);
    _232 += (_232 + 5ul);
    _232 += uint64_t(_23._m0._m0 + uint(_232));
    _232 += uint64_t(uint(_23._m0._m1 + int(_232)));
    _232 += uint64_t(clamp(_23._m0._m2 + float(_232), 0.0, 18446742974197923840.0));
    _232 += u64vec3(_23._m0._m3).z;
    _232 += uint64_t(_23._m0._m7);
    _232 += u64vec2(_23._m0._m8).y;
    _232 += u64vec3(_23._m0._m9).z;
    _232 += u64vec4(_23._m0._m10).w;
    _32._m0._m3 = _23._m0._m3 + _26._m0._m3;
    _32._m0._m4 = _23._m0._m4 + _26._m0._m4;
    _32._m0._m5 = _23._m0._m5 + _26._m0._m5;
    _32._m0._m6 = _23._m0._m6 + _26._m0._m6;
    _35._m0._m0 = _29._m0._m0;
    uint64_t _333 = _232;
    _232 += _333;
    _232 += min(max(_232, _232), _232);
    u64vec2 _344 = u64vec2(_232);
    u64vec2 _346 = u64vec2(_232);
    _232 += ((0ul + (_344.x * _346.x)) + (_344.y * _346.y));
    _232 += max(_232, _232);
    _232 += min(_232, _232);
    return _232;
}

int64_t _40(int64_t _39)
{
    int64_t _57 = 20l;
    _57 += ((31l - 1002003004005006l) + (-9223372036854775807l));
    _57 += (_57 + 5l);
    _57 += int64_t(int(_23._m0._m0 + uint(_57)));
    _57 += int64_t(_23._m0._m1 + int(_57));
    _57 += int64_t(clamp(_23._m0._m2 + float(_57), -9223372036854775808.0, 9223371487098961920.0));
    _57 += i64vec3(_23._m0._m7).z;
    _57 += int64_t(_23._m0._m3);
    _57 += i64vec2(_23._m0._m4).y;
    _57 += i64vec3(_23._m0._m5).z;
    _57 += i64vec4(_23._m0._m6).w;
    _57 += int64_t(0x8000000000000000ul);
    _32._m0._m7 = _23._m0._m7 + _26._m0._m7;
    _32._m0._m8 = _23._m0._m8 + _26._m0._m8;
    _32._m0._m9 = _23._m0._m9 + _26._m0._m9;
    _32._m0._m10 = _23._m0._m10 + _26._m0._m10;
    _35._m0._m1 = _29._m0._m1;
    _57 += abs(_57);
    _57 += min(max(_57, _57), _57);
    i64vec2 _192 = i64vec2(_57);
    i64vec2 _194 = i64vec2(_57);
    _57 += ((0l + (_192.x * _194.x)) + (_192.y * _194.y));
    _57 += max(_57, _57);
    _57 += min(_57, _57);
    _57 += sign(_57);
    return _57;
}

void main()
{
    uint64_t _379 = _222(67ul);
    int64_t _380 = _40(60l);
    _32._m0._m11 = _379 + uint64_t(_380);
}


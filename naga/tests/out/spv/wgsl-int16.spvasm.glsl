#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_subgroup_extended_types_int16 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _14
{
    uint _m0;
    int _m1;
    float _m2;
    uint16_t _m3;
    u16vec2 _m4;
    u16vec3 _m5;
    u16vec4 _m6;
    int16_t _m7;
    i16vec2 _m8;
    i16vec3 _m9;
    i16vec4 _m10;
    uint16_t _m11;
};

struct _18
{
    uint16_t _m0[2];
    int16_t _m1[2];
};

layout(set = 0, binding = 0, std140) uniform _27_26
{
    _14 _m0;
} _26;

layout(set = 0, binding = 1, std430) readonly buffer _30_29
{
    _14 _m0;
} _29;

layout(set = 0, binding = 2, std430) readonly buffer _33_32
{
    _18 _m0;
} _32;

layout(set = 0, binding = 3, std430) buffer _36_35
{
    _14 _m0;
} _35;

layout(set = 0, binding = 4, std430) buffer _39_38
{
    _18 _m0;
} _38;

int16_t _24 = 1s;
shared uint16_t _41;

uint16_t _240(uint16_t _242, uint16_t _243)
{
    return _242 / ((_243 == 0us) ? 1us : _243);
}

uint16_t _249(uint16_t _250, uint16_t _251)
{
    return _250 % ((_251 == 0us) ? 1us : _251);
}

uint16_t _258(uint16_t _257)
{
    uint16_t _271 = 20us;
    _271 += 5us;
    _271 += uint16_t(_26._m0._m0);
    _271 += uint16_t(_26._m0._m1);
    _271 += u16vec3(_26._m0._m3).z;
    _35._m0._m3 = _26._m0._m3 + _29._m0._m3;
    _35._m0._m4 = _26._m0._m4 + _29._m0._m4;
    _35._m0._m5 = _26._m0._m5 + _29._m0._m5;
    _35._m0._m6 = _26._m0._m6 + _29._m0._m6;
    _38._m0._m0 = _32._m0._m0;
    uint16_t _332 = _271;
    _271 = _332;
    _271 = max(_271, _271);
    _271 = min(_271, _271);
    _271 = min(max(_271, _271), _271);
    _271 -= 1us;
    _271 *= 2us;
    _271 = _240(_271, 3us);
    _271 = _249(_271, 4us);
    _271 &= 255us;
    _271 |= 16us;
    _271 ^= 1us;
    _35._m0._m0 = uint(_271);
    _35._m0._m1 = int(int16_t(_271));
    _35._m0._m2 = float(_271);
    _271 = uint16_t(_35._m0._m0);
    return _271;
}

int16_t _43(int16_t _45, int16_t _46)
{
    return _45 / (((_46 == 0s) || ((_45 == (-32768s)) && (_46 == (-1s)))) ? 1s : _46);
}

int16_t _59(int16_t _60, int16_t _61)
{
    return _60 - (((_61 == 0s) || ((_60 == (-32768s)) && (_61 == (-1s)))) ? 1s : _61) * (_60 / (((_61 == 0s) || ((_60 == (-32768s)) && (_61 == (-1s)))) ? 1s : _61));
}

int16_t _72(int16_t _71)
{
    int16_t _94 = 20s;
    int16_t _96[4] = int16_t[](1s, 2s, 3s, 4s);
    _94 += 5s;
    _94 += int16_t(_26._m0._m0);
    _94 += int16_t(_26._m0._m1);
    _94 += i16vec3(_26._m0._m7).z;
    _35._m0._m7 = _26._m0._m7 + _29._m0._m7;
    _35._m0._m8 = _26._m0._m8 + _29._m0._m8;
    _35._m0._m9 = _26._m0._m9 + _29._m0._m9;
    _35._m0._m10 = _26._m0._m10 + _29._m0._m10;
    _38._m0._m1 = _32._m0._m1;
    _94 = abs(_94);
    _94 = max(_94, _94);
    _94 = min(_94, _94);
    _94 = min(max(_94, _94), _94);
    _94 = sign(_94);
    _94 -= 1s;
    _94 *= 2s;
    _94 = _43(_94, 3s);
    _94 = _59(_94, 4s);
    _94 &= 255s;
    _94 |= 16s;
    _94 ^= 1s;
    _94 = _94 << 2u;
    _94 = _94 >> 1u;
    _94 = -_94;
    _94 = (_94 < 0s) ? 2s : 1s;
    _96[0u] = _94;
    _94 = _96[1u];
    _94 = _96[1u];
    _35._m0._m0 = uint(uint16_t(_94));
    _35._m0._m1 = int(_94);
    _35._m0._m2 = float(_94);
    _94 = int16_t(_35._m0._m0);
    _94 = int16_t(uint16_t(_94));
    _35._m0._m8 = (_26._m0._m8 + _26._m0._m8) * i16vec2(2);
    return _94;
}

void main()
{
    int16_t _384 = 0s;
    uint16_t _386 = 0us;
    if (gl_LocalInvocationIndex == 0u)
    {
        _41 = 0us;
    }
    barrier();
    _41 = 0us;
    uint16_t _397 = _258(67us);
    int16_t _398 = _72(60s);
    _35._m0._m11 = _397 + uint16_t(_398);
    _384 = int16_t(gl_SubgroupInvocationID);
    _384 = subgroupAdd(_384);
    _384 = subgroupMul(_384);
    _384 = subgroupMin(_384);
    _384 = subgroupMax(_384);
    _384 = subgroupExclusiveAdd(_384);
    _384 = subgroupInclusiveAdd(_384);
    _384 = subgroupBroadcastFirst(_384);
    _384 = subgroupShuffle(_384, 4u);
    _386 = uint16_t(gl_SubgroupInvocationID);
    _386 = subgroupAdd(_386);
    _386 = subgroupMin(_386);
    _386 = subgroupMax(_386);
    _35._m0._m7 = _384;
    _35._m0._m3 = _386;
}


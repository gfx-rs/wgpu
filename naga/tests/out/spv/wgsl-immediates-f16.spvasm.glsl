#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _5
{
    float16_t _m0;
    int _m1;
};

struct _10
{
    float16_t _m0;
    float16_t _m1;
    f16vec3 _m2;
    f16mat2x3 _m3;
    _5 _m4;
    float _m5;
    uint _m6;
};

struct _12
{
    uint _m0;
    uint _m1;
    uint _m2;
    uint _m3;
    uint _m4;
    uint _m5;
    uint _m6;
    uint _m7;
    uint _m8;
    uint _m9;
    uint _m10;
    uint _m11;
};

layout(set = 0, binding = 0, std430) buffer _18_17
{
    float _m0[];
} _17;

layout(push_constant, std430) uniform _15_14
{
    _12 _m0;
} _14;

void main()
{
    f16vec2 _35 = unpackFloat2x16(_14._m0._m0);
    _17._m0[0u] = float(_10(_35.x, _35.y, f16vec3(unpackFloat2x16(_14._m0._m2), unpackFloat2x16(_14._m0._m3).x), f16mat2x3(f16vec3(unpackFloat2x16(_14._m0._m4), unpackFloat2x16(_14._m0._m5).x), f16vec3(unpackFloat2x16(_14._m0._m6), unpackFloat2x16(_14._m0._m7).x)), _5(unpackFloat2x16(_14._m0._m8).x, int(_14._m0._m9)), uintBitsToFloat(_14._m0._m10), _14._m0._m11)._m0);
    _17._m0[1u] = float(unpackFloat2x16(_14._m0._m0).y);
    _17._m0[2u] = float(unpackFloat2x16(_14._m0._m3).x);
    _17._m0[3u] = float(f16vec3(unpackFloat2x16(_14._m0._m2), unpackFloat2x16(_14._m0._m3).x)[gl_LocalInvocationIndex]);
    _17._m0[4u] = float(f16vec3(unpackFloat2x16(_14._m0._m6), unpackFloat2x16(_14._m0._m7).x).y);
    f16mat2x3 _158 = f16mat2x3(f16vec3(unpackFloat2x16(_14._m0._m4), unpackFloat2x16(_14._m0._m5).x), f16vec3(unpackFloat2x16(_14._m0._m6), unpackFloat2x16(_14._m0._m7).x));
    _17._m0[5u] = float(_158[gl_LocalInvocationIndex].z);
    _17._m0[6u] = float(unpackFloat2x16(_14._m0._m8).x) + float(int(_14._m0._m9));
    _17._m0[7u] = uintBitsToFloat(_14._m0._m10) + float(_14._m0._m11);
}


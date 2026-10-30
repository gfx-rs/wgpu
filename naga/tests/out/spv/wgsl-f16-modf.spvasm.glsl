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

struct _7
{
    float16_t _m0;
    float16_t _m1;
};

struct _9
{
    f16vec2 _m0;
    f16vec2 _m1;
};

struct _11
{
    f16vec3 _m0;
    f16vec3 _m1;
};

struct _13
{
    f16vec4 _m0;
    f16vec4 _m1;
};

layout(set = 0, binding = 0, std430) buffer _15_14
{
    float16_t _m0[4];
} _14;

void main()
{
    float16_t _26 = _14._m0[0u];
    _7 _27;
    _27._m0 = modf(_26, _27._m1);
    _14._m0[0u] = _27._m0;
    _9 _31;
    _31._m0 = modf(f16vec2(_26), _31._m1);
    _14._m0[1u] = _31._m1.x;
    _11 _37;
    _37._m0 = modf(f16vec3(_26), _37._m1);
    _14._m0[2u] = _37._m0.y;
    _13 _43;
    _43._m0 = modf(f16vec4(_26), _43._m1);
    _14._m0[3u] = _43._m1.z;
}


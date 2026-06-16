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

struct _19
{
    uint _m0;
    int _m1;
    float _m2;
    float16_t _m3;
    f16vec2 _m4;
    f16vec3 _m5;
    f16vec4 _m6;
    float16_t _m7;
    f16mat2 _m8;
    f16mat2x3 _m9;
    f16mat2x4 _m10;
    f16mat3x2 _m11;
    f16mat3 _m12;
    f16mat3x4 _m13;
    f16mat4x2 _m14;
    f16mat4x3 _m15;
    f16mat4 _m16;
};

struct _22
{
    float16_t _m0[2];
};

struct _23
{
    float16_t _m0;
    float16_t _m1;
    f16vec3 _m2;
    float16_t _m3;
    float16_t _m4;
    uint _m5;
};

struct _24
{
    uint _m0;
    int _m1;
    float _m2;
    float16_t _m3;
    f16vec2 _m4;
    f16vec3 _m5;
    f16vec4 _m6;
    float16_t _m7;
    f16vec2 _m8;
    f16vec2 _m9;
    f16mat2x3 _m10;
    f16mat2x4 _m11;
    f16vec2 _m12;
    f16vec2 _m13;
    f16vec2 _m14;
    f16mat3 _m15;
    f16mat3x4 _m16;
    f16vec2 _m17;
    f16vec2 _m18;
    f16vec2 _m19;
    f16vec2 _m20;
    f16mat4x3 _m21;
    f16mat4 _m22;
};

layout(set = 0, binding = 0, std140) uniform _30_29
{
    _24 _m0;
} _29;

layout(set = 0, binding = 1, std430) readonly buffer _33_32
{
    _19 _m0;
} _32;

layout(set = 0, binding = 2, std430) readonly buffer _36_35
{
    _22 _m0;
} _35;

layout(set = 0, binding = 3, std430) buffer _39_38
{
    _19 _m0;
} _38;

layout(set = 0, binding = 4, std430) buffer _42_41
{
    _22 _m0;
} _41;

float16_t _27 = float16_t(1.0);

float16_t _46(float16_t _45)
{
    _23 _65 = _23(float16_t(0.0), float16_t(0.0), f16vec3(float16_t(0.0)), float16_t(0.0), float16_t(0.0), 0u);
    float16_t _68 = float16_t(15.203125);
    _68 += float16_t(-33344.0);
    _68 += (_68 + float16_t(5.0));
    _68 += float16_t(_29._m0._m2 + float(_68));
    _68 += f16vec3(_29._m0._m3).z;
    _38._m0._m1 = 65504;
    _38._m0._m1 = -65504;
    _38._m0._m0 = 65504u;
    _38._m0._m0 = 0u;
    _38._m0._m2 = 65504.0;
    _38._m0._m2 = -65504.0;
    _38._m0._m3 = _29._m0._m3 + _32._m0._m3;
    _38._m0._m4 = _29._m0._m4 + _32._m0._m4;
    _38._m0._m5 = _29._m0._m5 + _32._m0._m5;
    _38._m0._m6 = _29._m0._m6 + _32._m0._m6;
    f16mat2 _147 = f16mat2(_29._m0._m8, _29._m0._m9);
    _38._m0._m8 = f16mat2(_147[0] + _32._m0._m8[0], _147[1] + _32._m0._m8[1]);
    _38._m0._m9 = f16mat2x3(_29._m0._m10[0] + _32._m0._m9[0], _29._m0._m10[1] + _32._m0._m9[1]);
    _38._m0._m10 = f16mat2x4(_29._m0._m11[0] + _32._m0._m10[0], _29._m0._m11[1] + _32._m0._m10[1]);
    f16mat3x2 _199 = f16mat3x2(_29._m0._m12, _29._m0._m13, _29._m0._m14);
    _38._m0._m11 = f16mat3x2(_199[0] + _32._m0._m11[0], _199[1] + _32._m0._m11[1], _199[2] + _32._m0._m11[2]);
    _38._m0._m12 = f16mat3(_29._m0._m15[0] + _32._m0._m12[0], _29._m0._m15[1] + _32._m0._m12[1], _29._m0._m15[2] + _32._m0._m12[2]);
    _38._m0._m13 = f16mat3x4(_29._m0._m16[0] + _32._m0._m13[0], _29._m0._m16[1] + _32._m0._m13[1], _29._m0._m16[2] + _32._m0._m13[2]);
    f16mat4x2 _263 = f16mat4x2(_29._m0._m17, _29._m0._m18, _29._m0._m19, _29._m0._m20);
    _38._m0._m14 = f16mat4x2(_263[0] + _32._m0._m14[0], _263[1] + _32._m0._m14[1], _263[2] + _32._m0._m14[2], _263[3] + _32._m0._m14[3]);
    _38._m0._m15 = f16mat4x3(_29._m0._m21[0] + _32._m0._m15[0], _29._m0._m21[1] + _32._m0._m15[1], _29._m0._m21[2] + _32._m0._m15[2], _29._m0._m21[3] + _32._m0._m15[3]);
    _38._m0._m16 = f16mat4(_29._m0._m22[0] + _32._m0._m16[0], _29._m0._m22[1] + _32._m0._m16[1], _29._m0._m22[2] + _32._m0._m16[2], _29._m0._m22[3] + _32._m0._m16[3]);
    _41._m0._m0 = _35._m0._m0;
    _68 += abs(_68);
    _68 += clamp(_68, _68, _68);
    _68 += dot(f16vec2(_68), f16vec2(_68));
    _68 += max(_68, _68);
    _68 += min(_68, _68);
    _68 += sign(_68);
    _68 += float16_t(1.0);
    _38._m0._m4 = f16vec2(vec2(_29._m0._m4));
    _38._m0._m5 = f16vec3(vec3(_29._m0._m5));
    _38._m0._m6 = f16vec4(vec4(_29._m0._m6));
    f16mat2 _381 = f16mat2(_29._m0._m8, _29._m0._m9);
    mat2 _387 = mat2(vec2(_381[0]), vec2(_381[1]));
    _38._m0._m8 = f16mat2(f16vec2(_387[0]), f16vec2(_387[1]));
    mat2x3 _401 = mat2x3(vec3(_29._m0._m10[0]), vec3(_29._m0._m10[1]));
    _38._m0._m9 = f16mat2x3(f16vec3(_401[0]), f16vec3(_401[1]));
    mat2x4 _415 = mat2x4(vec4(_29._m0._m11[0]), vec4(_29._m0._m11[1]));
    _38._m0._m10 = f16mat2x4(f16vec4(_415[0]), f16vec4(_415[1]));
    f16mat3x2 _428 = f16mat3x2(_29._m0._m12, _29._m0._m13, _29._m0._m14);
    mat3x2 _436 = mat3x2(vec2(_428[0]), vec2(_428[1]), vec2(_428[2]));
    _38._m0._m11 = f16mat3x2(f16vec2(_436[0]), f16vec2(_436[1]), f16vec2(_436[2]));
    mat3 _454 = mat3(vec3(_29._m0._m15[0]), vec3(_29._m0._m15[1]), vec3(_29._m0._m15[2]));
    _38._m0._m12 = f16mat3(f16vec3(_454[0]), f16vec3(_454[1]), f16vec3(_454[2]));
    mat3x4 _472 = mat3x4(vec4(_29._m0._m16[0]), vec4(_29._m0._m16[1]), vec4(_29._m0._m16[2]));
    _38._m0._m13 = f16mat3x4(f16vec4(_472[0]), f16vec4(_472[1]), f16vec4(_472[2]));
    f16mat4x2 _489 = f16mat4x2(_29._m0._m17, _29._m0._m18, _29._m0._m19, _29._m0._m20);
    mat4x2 _499 = mat4x2(vec2(_489[0]), vec2(_489[1]), vec2(_489[2]), vec2(_489[3]));
    _38._m0._m14 = f16mat4x2(f16vec2(_499[0]), f16vec2(_499[1]), f16vec2(_499[2]), f16vec2(_499[3]));
    mat4x3 _521 = mat4x3(vec3(_29._m0._m21[0]), vec3(_29._m0._m21[1]), vec3(_29._m0._m21[2]), vec3(_29._m0._m21[3]));
    _38._m0._m15 = f16mat4x3(f16vec3(_521[0]), f16vec3(_521[1]), f16vec3(_521[2]), f16vec3(_521[3]));
    mat4 _543 = mat4(vec4(_29._m0._m22[0]), vec4(_29._m0._m22[1]), vec4(_29._m0._m22[2]), vec4(_29._m0._m22[3]));
    _38._m0._m16 = f16mat4(f16vec4(_543[0]), f16vec4(_543[1]), f16vec4(_543[2]), f16vec4(_543[3]));
    return _68;
}

void main()
{
    float16_t _565 = _46(float16_t(2.0));
    _38._m0._m7 = _565;
}


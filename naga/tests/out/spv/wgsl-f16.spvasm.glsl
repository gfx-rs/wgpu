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
    f16vec3 _m10;
    f16vec3 _m11;
    f16vec4 _m12;
    f16vec4 _m13;
    f16vec2 _m14;
    f16vec2 _m15;
    f16vec2 _m16;
    f16vec3 _m17;
    f16vec3 _m18;
    f16vec3 _m19;
    f16vec4 _m20;
    f16vec4 _m21;
    f16vec4 _m22;
    f16vec2 _m23;
    f16vec2 _m24;
    f16vec2 _m25;
    f16vec2 _m26;
    f16vec3 _m27;
    f16vec3 _m28;
    f16vec3 _m29;
    f16vec3 _m30;
    f16vec4 _m31;
    f16vec4 _m32;
    f16vec4 _m33;
    f16vec4 _m34;
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
    _23 _64 = _23(float16_t(0.0), float16_t(0.0), f16vec3(float16_t(0.0)), float16_t(0.0), float16_t(0.0), 0u);
    float16_t _67 = float16_t(15.203125);
    _67 += float16_t(-33344.0);
    _67 += (_67 + float16_t(5.0));
    _67 += float16_t(_29._m0._m2 + float(_67));
    _67 += f16vec3(_29._m0._m3).z;
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
    f16mat2 _145 = f16mat2(_29._m0._m8, _29._m0._m9);
    _38._m0._m8 = f16mat2(_145[0] + _32._m0._m8[0], _145[1] + _32._m0._m8[1]);
    f16mat2x3 _163 = f16mat2x3(_29._m0._m10, _29._m0._m11);
    _38._m0._m9 = f16mat2x3(_163[0] + _32._m0._m9[0], _163[1] + _32._m0._m9[1]);
    f16mat2x4 _181 = f16mat2x4(_29._m0._m12, _29._m0._m13);
    _38._m0._m10 = f16mat2x4(_181[0] + _32._m0._m10[0], _181[1] + _32._m0._m10[1]);
    f16mat3x2 _202 = f16mat3x2(_29._m0._m14, _29._m0._m15, _29._m0._m16);
    _38._m0._m11 = f16mat3x2(_202[0] + _32._m0._m11[0], _202[1] + _32._m0._m11[1], _202[2] + _32._m0._m11[2]);
    f16mat3 _226 = f16mat3(_29._m0._m17, _29._m0._m18, _29._m0._m19);
    _38._m0._m12 = f16mat3(_226[0] + _32._m0._m12[0], _226[1] + _32._m0._m12[1], _226[2] + _32._m0._m12[2]);
    f16mat3x4 _250 = f16mat3x4(_29._m0._m20, _29._m0._m21, _29._m0._m22);
    _38._m0._m13 = f16mat3x4(_250[0] + _32._m0._m13[0], _250[1] + _32._m0._m13[1], _250[2] + _32._m0._m13[2]);
    f16mat4x2 _277 = f16mat4x2(_29._m0._m23, _29._m0._m24, _29._m0._m25, _29._m0._m26);
    _38._m0._m14 = f16mat4x2(_277[0] + _32._m0._m14[0], _277[1] + _32._m0._m14[1], _277[2] + _32._m0._m14[2], _277[3] + _32._m0._m14[3]);
    f16mat4x3 _307 = f16mat4x3(_29._m0._m27, _29._m0._m28, _29._m0._m29, _29._m0._m30);
    _38._m0._m15 = f16mat4x3(_307[0] + _32._m0._m15[0], _307[1] + _32._m0._m15[1], _307[2] + _32._m0._m15[2], _307[3] + _32._m0._m15[3]);
    f16mat4 _337 = f16mat4(_29._m0._m31, _29._m0._m32, _29._m0._m33, _29._m0._m34);
    _38._m0._m16 = f16mat4(_337[0] + _32._m0._m16[0], _337[1] + _32._m0._m16[1], _337[2] + _32._m0._m16[2], _337[3] + _32._m0._m16[3]);
    _41._m0._m0 = _35._m0._m0;
    _67 += abs(_67);
    _67 += clamp(_67, _67, _67);
    _67 += dot(f16vec2(_67), f16vec2(_67));
    _67 += max(_67, _67);
    _67 += min(_67, _67);
    _67 += sign(_67);
    _67 += float16_t(1.0);
    _38._m0._m4 = f16vec2(vec2(_29._m0._m4));
    _38._m0._m5 = f16vec3(vec3(_29._m0._m5));
    _38._m0._m6 = f16vec4(vec4(_29._m0._m6));
    f16mat2 _413 = f16mat2(_29._m0._m8, _29._m0._m9);
    mat2 _419 = mat2(vec2(_413[0]), vec2(_413[1]));
    _38._m0._m8 = f16mat2(f16vec2(_419[0]), f16vec2(_419[1]));
    f16mat2x3 _430 = f16mat2x3(_29._m0._m10, _29._m0._m11);
    mat2x3 _436 = mat2x3(vec3(_430[0]), vec3(_430[1]));
    _38._m0._m9 = f16mat2x3(f16vec3(_436[0]), f16vec3(_436[1]));
    f16mat2x4 _447 = f16mat2x4(_29._m0._m12, _29._m0._m13);
    mat2x4 _453 = mat2x4(vec4(_447[0]), vec4(_447[1]));
    _38._m0._m10 = f16mat2x4(f16vec4(_453[0]), f16vec4(_453[1]));
    f16mat3x2 _466 = f16mat3x2(_29._m0._m14, _29._m0._m15, _29._m0._m16);
    mat3x2 _474 = mat3x2(vec2(_466[0]), vec2(_466[1]), vec2(_466[2]));
    _38._m0._m11 = f16mat3x2(f16vec2(_474[0]), f16vec2(_474[1]), f16vec2(_474[2]));
    f16mat3 _489 = f16mat3(_29._m0._m17, _29._m0._m18, _29._m0._m19);
    mat3 _497 = mat3(vec3(_489[0]), vec3(_489[1]), vec3(_489[2]));
    _38._m0._m12 = f16mat3(f16vec3(_497[0]), f16vec3(_497[1]), f16vec3(_497[2]));
    f16mat3x4 _512 = f16mat3x4(_29._m0._m20, _29._m0._m21, _29._m0._m22);
    mat3x4 _520 = mat3x4(vec4(_512[0]), vec4(_512[1]), vec4(_512[2]));
    _38._m0._m13 = f16mat3x4(f16vec4(_520[0]), f16vec4(_520[1]), f16vec4(_520[2]));
    f16mat4x2 _537 = f16mat4x2(_29._m0._m23, _29._m0._m24, _29._m0._m25, _29._m0._m26);
    mat4x2 _547 = mat4x2(vec2(_537[0]), vec2(_537[1]), vec2(_537[2]), vec2(_537[3]));
    _38._m0._m14 = f16mat4x2(f16vec2(_547[0]), f16vec2(_547[1]), f16vec2(_547[2]), f16vec2(_547[3]));
    f16mat4x3 _566 = f16mat4x3(_29._m0._m27, _29._m0._m28, _29._m0._m29, _29._m0._m30);
    mat4x3 _576 = mat4x3(vec3(_566[0]), vec3(_566[1]), vec3(_566[2]), vec3(_566[3]));
    _38._m0._m15 = f16mat4x3(f16vec3(_576[0]), f16vec3(_576[1]), f16vec3(_576[2]), f16vec3(_576[3]));
    f16mat4 _595 = f16mat4(_29._m0._m31, _29._m0._m32, _29._m0._m33, _29._m0._m34);
    mat4 _605 = mat4(vec4(_595[0]), vec4(_595[1]), vec4(_595[2]), vec4(_595[3]));
    _38._m0._m16 = f16mat4(f16vec4(_605[0]), f16vec4(_605[1]), f16vec4(_605[2]), f16vec4(_605[3]));
    return _67;
}

void main()
{
    float16_t _627 = _46(float16_t(2.0));
    _38._m0._m7 = _627;
}


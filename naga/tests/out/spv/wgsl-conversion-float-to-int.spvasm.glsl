#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void _24()
{
    int64_t _73 = 9223372036854774784l;
    uint _70 = 0u;
    uint64_t _67 = 18446744073709549568ul;
    int64_t _64 = int64_t(0x8000000000000000ul);
    int64_t _61 = 9223371487098961920l;
    uint _58 = 0u;
    uint64_t _55 = 65504ul;
    int64_t _50 = -65504l;
    int _46 = 65504;
    uint64_t _75 = 18446744073709549568ul;
    int64_t _72 = int64_t(0x8000000000000000ul);
    int _69 = 2147483647;
    uint64_t _66 = 0ul;
    uint64_t _63 = 18446742974197923840ul;
    int64_t _60 = int64_t(0x8000000000000000ul);
    int _57 = 2147483520;
    uint64_t _53 = 0ul;
    uint _49 = 65504u;
    int _44 = -65504;
    uint64_t _74 = 0ul;
    uint _71 = 4294967295u;
    int _68 = int(0x80000000);
    int64_t _65 = 9223372036854774784l;
    uint64_t _62 = 0ul;
    uint _59 = 4294967040u;
    int _56 = int(0x80000000);
    int64_t _52 = 65504l;
    uint _47 = 0u;
}

int _79(float16_t _78)
{
    return int(clamp(_78, float16_t(-65504.0), float16_t(65504.0)));
}

uint _86(float16_t _85)
{
    return uint(clamp(_85, float16_t(0.0), float16_t(65504.0)));
}

int64_t _94(float16_t _93)
{
    return int64_t(clamp(_93, float16_t(-65504.0), float16_t(65504.0)));
}

uint64_t _101(float16_t _100)
{
    return uint64_t(clamp(_100, float16_t(0.0), float16_t(65504.0)));
}

int _108(float _107)
{
    return int(clamp(_107, -2147483648.0, 2147483520.0));
}

uint _117(float _116)
{
    return uint(clamp(_116, 0.0, 4294967040.0));
}

int64_t _126(float _125)
{
    return int64_t(clamp(_125, -9223372036854775808.0, 9223371487098961920.0));
}

uint64_t _135(float _134)
{
    return uint64_t(clamp(_134, 0.0, 18446742974197923840.0));
}

int _143(double _142)
{
    return int(clamp(_142, -2147483648.0lf, 2147483647.0lf));
}

uint _152(double _151)
{
    return uint(clamp(_151, 0.0lf, 4294967295.0lf));
}

int64_t _161(double _160)
{
    return int64_t(clamp(_160, -9223372036854775808.0lf, 9223372036854774784.0lf));
}

uint64_t _170(double _169)
{
    return uint64_t(clamp(_169, 0.0lf, 18446744073709549568.0lf));
}

ivec2 _178(f16vec2 _177)
{
    return ivec2(clamp(_177, f16vec2(float16_t(-65504.0)), f16vec2(float16_t(65504.0))));
}

uvec2 _187(f16vec2 _186)
{
    return uvec2(clamp(_186, f16vec2(float16_t(0.0)), f16vec2(float16_t(65504.0))));
}

i64vec2 _195(f16vec2 _194)
{
    return i64vec2(clamp(_194, f16vec2(float16_t(-65504.0)), f16vec2(float16_t(65504.0))));
}

u64vec2 _202(f16vec2 _201)
{
    return u64vec2(clamp(_201, f16vec2(float16_t(0.0)), f16vec2(float16_t(65504.0))));
}

ivec2 _209(vec2 _208)
{
    return ivec2(clamp(_208, vec2(-2147483648.0), vec2(2147483520.0)));
}

uvec2 _218(vec2 _217)
{
    return uvec2(clamp(_217, vec2(0.0), vec2(4294967040.0)));
}

i64vec2 _227(vec2 _226)
{
    return i64vec2(clamp(_226, vec2(-9223372036854775808.0), vec2(9223371487098961920.0)));
}

u64vec2 _236(vec2 _235)
{
    return u64vec2(clamp(_235, vec2(0.0), vec2(18446742974197923840.0)));
}

ivec2 _244(dvec2 _243)
{
    return ivec2(clamp(_243, dvec2(-2147483648.0lf), dvec2(2147483647.0lf)));
}

uvec2 _253(dvec2 _252)
{
    return uvec2(clamp(_252, dvec2(0.0lf), dvec2(4294967295.0lf)));
}

i64vec2 _262(dvec2 _261)
{
    return i64vec2(clamp(_261, dvec2(-9223372036854775808.0lf), dvec2(9223372036854774784.0lf)));
}

u64vec2 _271(dvec2 _270)
{
    return u64vec2(clamp(_270, dvec2(0.0lf), dvec2(18446744073709549568.0lf)));
}

void main()
{
    _24();
}


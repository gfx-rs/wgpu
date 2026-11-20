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

void main()
{
    uvec2 _51 = uvec2(0u);
    uint _45 = 0u;
    f16vec2 _39 = f16vec2(float16_t(1.0), float16_t(2.0));
    vec2 _53 = vec2(0.0);
    float _47 = 0.0;
    f16vec4 _41 = f16vec4(float16_t(1.0), float16_t(2.0), float16_t(3.0), float16_t(4.0));
    ivec2 _49 = ivec2(0);
    int _43 = 0;
    _43 = (_39);
    _45 = packFloat2x16(_39);
    _47 = uintBitsToFloat(packFloat2x16(_39));
    _49 = (_41);
    _51 = (_41);
    _53 = (_41);
    _39 = (_43);
    _39 = unpackFloat2x16(_45);
    _39 = unpackFloat2x16(floatBitsToUint(_47));
    _41 = (_49);
    _41 = (_51);
    _41 = (_53);
    _39 = _39;
    _45 = _45;
    _45 = 4294967295u;
    _47 = intBitsToFloat(1);
    _43 = floatBitsToInt(1.0);
    _39 = (1);
    _39 = f16vec2(float16_t(1.0), float16_t(2.0));
    _41 = (ivec2(1, 2));
    _51 = uvec2(4294967295u, 1u);
}


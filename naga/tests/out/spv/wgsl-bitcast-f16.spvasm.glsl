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
    uvec2 _46 = uvec2(0u);
    uint _40 = 0u;
    f16vec2 _34 = f16vec2(float16_t(1.0), float16_t(2.0));
    vec2 _48 = vec2(0.0);
    float _42 = 0.0;
    f16vec4 _36 = f16vec4(float16_t(1.0), float16_t(2.0), float16_t(3.0), float16_t(4.0));
    ivec2 _44 = ivec2(0);
    int _38 = 0;
    _38 = (_34);
    _40 = packFloat2x16(_34);
    _42 = uintBitsToFloat(packFloat2x16(_34));
    _44 = (_36);
    _46 = (_36);
    _48 = (_36);
    _34 = (_38);
    _34 = unpackFloat2x16(_40);
    _34 = unpackFloat2x16(floatBitsToUint(_42));
    _36 = (_44);
    _36 = (_46);
    _36 = (_48);
    _34 = _34;
    _40 = _40;
    _40 = 4294967295u;
    _42 = intBitsToFloat(1);
    _38 = floatBitsToInt(1.0);
    _34 = (1);
    _34 = f16vec2(float16_t(1.0), float16_t(2.0));
    _36 = (ivec2(1, 2));
    _46 = uvec2(4294967295u, 1u);
}


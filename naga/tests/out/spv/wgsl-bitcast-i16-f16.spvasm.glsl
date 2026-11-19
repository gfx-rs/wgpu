#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    u16vec4 _30 = u16vec4(1us, 2us, 3us, 4us);
    int16_t _24 = 1s;
    f16vec4 _28 = f16vec4(float16_t(1.0), float16_t(2.0), float16_t(3.0), float16_t(4.0));
    float16_t _22 = float16_t(1.0);
    uint16_t _26 = 2us;
    _24 = float16BitsToInt16(_22);
    _26 = float16BitsToUint16(_22);
    _22 = int16BitsToFloat16(_24);
    _22 = uint16BitsToFloat16(_26);
    _30 = float16BitsToUint16(_28);
    _28 = uint16BitsToFloat16(_30);
}


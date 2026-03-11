///////////////////////////////////////
// Entry point: "test_direct" (frag) //
///////////////////////////////////////
#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require

struct F16IO
{
    float16_t scalar_f16;
    float scalar_f32;
    f16vec2 vec2_f16;
    vec2 vec2_f32;
    f16vec3 vec3_f16;
    vec3 vec3_f32;
    f16vec4 vec4_f16;
    vec4 vec4_f32;
};

layout(location = 0) in float16_t scalar_f16;
layout(location = 1) in float scalar_f32;
layout(location = 2) in f16vec2 vec2_f16;
layout(location = 3) in vec2 vec2_f32;
layout(location = 4) in f16vec3 vec3_f16;
layout(location = 5) in vec3 vec3_f32;
layout(location = 6) in f16vec4 vec4_f16;
layout(location = 7) in vec4 vec4_f32;
layout(location = 0) out float16_t scalar_f16_1;
layout(location = 1) out float scalar_f32_1;
layout(location = 2) out f16vec2 vec2_f16_1;
layout(location = 3) out vec2 vec2_f32_1;
layout(location = 4) out f16vec3 vec3_f16_1;
layout(location = 5) out vec3 vec3_f32_1;
layout(location = 6) out f16vec4 vec4_f16_1;
layout(location = 7) out vec4 vec4_f32_1;

void main()
{
    F16IO _output = F16IO(float16_t(0.0), 0.0, f16vec2(float16_t(0.0)), vec2(0.0), f16vec3(float16_t(0.0)), vec3(0.0), f16vec4(float16_t(0.0)), vec4(0.0));
    _output.scalar_f16 = scalar_f16 + float16_t(1.0);
    _output.scalar_f32 = scalar_f32 + 1.0;
    _output.vec2_f16 = vec2_f16 + f16vec2(float16_t(1.0));
    _output.vec2_f32 = vec2_f32 + vec2(1.0);
    _output.vec3_f16 = vec3_f16 + f16vec3(float16_t(1.0));
    _output.vec3_f32 = vec3_f32 + vec3(1.0);
    _output.vec4_f16 = vec4_f16 + f16vec4(float16_t(1.0));
    _output.vec4_f32 = vec4_f32 + vec4(1.0);
    scalar_f16_1 = _output.scalar_f16;
    scalar_f32_1 = _output.scalar_f32;
    vec2_f16_1 = _output.vec2_f16;
    vec2_f32_1 = _output.vec2_f32;
    vec3_f16_1 = _output.vec3_f16;
    vec3_f32_1 = _output.vec3_f32;
    vec4_f16_1 = _output.vec4_f16;
    vec4_f32_1 = _output.vec4_f32;
}


///////////////////////////////////////
// Entry point: "test_struct" (frag) //
///////////////////////////////////////
#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require

struct F16IO
{
    float16_t scalar_f16;
    float scalar_f32;
    f16vec2 vec2_f16;
    vec2 vec2_f32;
    f16vec3 vec3_f16;
    vec3 vec3_f32;
    f16vec4 vec4_f16;
    vec4 vec4_f32;
};

layout(location = 0) in float16_t scalar_f16;
layout(location = 1) in float scalar_f32;
layout(location = 2) in f16vec2 vec2_f16;
layout(location = 3) in vec2 vec2_f32;
layout(location = 4) in f16vec3 vec3_f16;
layout(location = 5) in vec3 vec3_f32;
layout(location = 6) in f16vec4 vec4_f16;
layout(location = 7) in vec4 vec4_f32;
layout(location = 0) out float16_t scalar_f16_1;
layout(location = 1) out float scalar_f32_1;
layout(location = 2) out f16vec2 vec2_f16_1;
layout(location = 3) out vec2 vec2_f32_1;
layout(location = 4) out f16vec3 vec3_f16_1;
layout(location = 5) out vec3 vec3_f32_1;
layout(location = 6) out f16vec4 vec4_f16_1;
layout(location = 7) out vec4 vec4_f32_1;

void main()
{
    F16IO _output = F16IO(float16_t(0.0), 0.0, f16vec2(float16_t(0.0)), vec2(0.0), f16vec3(float16_t(0.0)), vec3(0.0), f16vec4(float16_t(0.0)), vec4(0.0));
    F16IO _111 = F16IO(scalar_f16, scalar_f32, vec2_f16, vec2_f32, vec3_f16, vec3_f32, vec4_f16, vec4_f32);
    _output.scalar_f16 = _111.scalar_f16 + float16_t(1.0);
    _output.scalar_f32 = _111.scalar_f32 + 1.0;
    _output.vec2_f16 = _111.vec2_f16 + f16vec2(float16_t(1.0));
    _output.vec2_f32 = _111.vec2_f32 + vec2(1.0);
    _output.vec3_f16 = _111.vec3_f16 + f16vec3(float16_t(1.0));
    _output.vec3_f32 = _111.vec3_f32 + vec3(1.0);
    _output.vec4_f16 = _111.vec4_f16 + f16vec4(float16_t(1.0));
    _output.vec4_f32 = _111.vec4_f32 + vec4(1.0);
    scalar_f16_1 = _output.scalar_f16;
    scalar_f32_1 = _output.scalar_f32;
    vec2_f16_1 = _output.vec2_f16;
    vec2_f32_1 = _output.vec2_f32;
    vec3_f16_1 = _output.vec3_f16;
    vec3_f32_1 = _output.vec3_f32;
    vec4_f16_1 = _output.vec4_f16;
    vec4_f32_1 = _output.vec4_f32;
}


///////////////////////////////////////////
// Entry point: "test_copy_input" (frag) //
///////////////////////////////////////////
#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require

struct F16IO
{
    float16_t scalar_f16;
    float scalar_f32;
    f16vec2 vec2_f16;
    vec2 vec2_f32;
    f16vec3 vec3_f16;
    vec3 vec3_f32;
    f16vec4 vec4_f16;
    vec4 vec4_f32;
};

layout(location = 0) in float16_t scalar_f16;
layout(location = 1) in float scalar_f32;
layout(location = 2) in f16vec2 vec2_f16;
layout(location = 3) in vec2 vec2_f32;
layout(location = 4) in f16vec3 vec3_f16;
layout(location = 5) in vec3 vec3_f32;
layout(location = 6) in f16vec4 vec4_f16;
layout(location = 7) in vec4 vec4_f32;
layout(location = 0) out float16_t scalar_f16_1;
layout(location = 1) out float scalar_f32_1;
layout(location = 2) out f16vec2 vec2_f16_1;
layout(location = 3) out vec2 vec2_f32_1;
layout(location = 4) out f16vec3 vec3_f16_1;
layout(location = 5) out vec3 vec3_f32_1;
layout(location = 6) out f16vec4 vec4_f16_1;
layout(location = 7) out vec4 vec4_f32_1;

void main()
{
    F16IO _input = F16IO(float16_t(0.0), 0.0, f16vec2(float16_t(0.0)), vec2(0.0), f16vec3(float16_t(0.0)), vec3(0.0), f16vec4(float16_t(0.0)), vec4(0.0));
    F16IO _output = F16IO(float16_t(0.0), 0.0, f16vec2(float16_t(0.0)), vec2(0.0), f16vec3(float16_t(0.0)), vec3(0.0), f16vec4(float16_t(0.0)), vec4(0.0));
    _input = F16IO(scalar_f16, scalar_f32, vec2_f16, vec2_f32, vec3_f16, vec3_f32, vec4_f16, vec4_f32);
    _output.scalar_f16 = _input.scalar_f16 + float16_t(1.0);
    _output.scalar_f32 = _input.scalar_f32 + 1.0;
    _output.vec2_f16 = _input.vec2_f16 + f16vec2(float16_t(1.0));
    _output.vec2_f32 = _input.vec2_f32 + vec2(1.0);
    _output.vec3_f16 = _input.vec3_f16 + f16vec3(float16_t(1.0));
    _output.vec3_f32 = _input.vec3_f32 + vec3(1.0);
    _output.vec4_f16 = _input.vec4_f16 + f16vec4(float16_t(1.0));
    _output.vec4_f32 = _input.vec4_f32 + vec4(1.0);
    scalar_f16_1 = _output.scalar_f16;
    scalar_f32_1 = _output.scalar_f32;
    vec2_f16_1 = _output.vec2_f16;
    vec2_f32_1 = _output.vec2_f32;
    vec3_f16_1 = _output.vec3_f16;
    vec3_f32_1 = _output.vec3_f32;
    vec4_f16_1 = _output.vec4_f16;
    vec4_f32_1 = _output.vec4_f32;
}


///////////////////////////////////////////////
// Entry point: "test_return_partial" (frag) //
///////////////////////////////////////////////
#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require

struct F16IO
{
    float16_t scalar_f16;
    float scalar_f32;
    f16vec2 vec2_f16;
    vec2 vec2_f32;
    f16vec3 vec3_f16;
    vec3 vec3_f32;
    f16vec4 vec4_f16;
    vec4 vec4_f32;
};

layout(location = 0) in float16_t scalar_f16;
layout(location = 1) in float scalar_f32;
layout(location = 2) in f16vec2 vec2_f16;
layout(location = 3) in vec2 vec2_f32;
layout(location = 4) in f16vec3 vec3_f16;
layout(location = 5) in vec3 vec3_f32;
layout(location = 6) in f16vec4 vec4_f16;
layout(location = 7) in vec4 vec4_f32;
layout(location = 0) out float16_t _264;

void main()
{
    F16IO _input = F16IO(float16_t(0.0), 0.0, f16vec2(float16_t(0.0)), vec2(0.0), f16vec3(float16_t(0.0)), vec3(0.0), f16vec4(float16_t(0.0)), vec4(0.0));
    _input = F16IO(scalar_f16, scalar_f32, vec2_f16, vec2_f32, vec3_f16, vec3_f32, vec4_f16, vec4_f32);
    _input.scalar_f16 = float16_t(0.0);
    _264 = _input.scalar_f16;
}


/////////////////////////////////////////////////
// Entry point: "test_component_access" (frag) //
/////////////////////////////////////////////////
#version 460
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require

struct F16IO
{
    float16_t scalar_f16;
    float scalar_f32;
    f16vec2 vec2_f16;
    vec2 vec2_f32;
    f16vec3 vec3_f16;
    vec3 vec3_f32;
    f16vec4 vec4_f16;
    vec4 vec4_f32;
};

layout(location = 0) in float16_t scalar_f16;
layout(location = 1) in float scalar_f32;
layout(location = 2) in f16vec2 vec2_f16;
layout(location = 3) in vec2 vec2_f32;
layout(location = 4) in f16vec3 vec3_f16;
layout(location = 5) in vec3 vec3_f32;
layout(location = 6) in f16vec4 vec4_f16;
layout(location = 7) in vec4 vec4_f32;
layout(location = 0) out float16_t scalar_f16_1;
layout(location = 1) out float scalar_f32_1;
layout(location = 2) out f16vec2 vec2_f16_1;
layout(location = 3) out vec2 vec2_f32_1;
layout(location = 4) out f16vec3 vec3_f16_1;
layout(location = 5) out vec3 vec3_f32_1;
layout(location = 6) out f16vec4 vec4_f16_1;
layout(location = 7) out vec4 vec4_f32_1;

void main()
{
    F16IO _output = F16IO(float16_t(0.0), 0.0, f16vec2(float16_t(0.0)), vec2(0.0), f16vec3(float16_t(0.0)), vec3(0.0), f16vec4(float16_t(0.0)), vec4(0.0));
    F16IO _274 = F16IO(scalar_f16, scalar_f32, vec2_f16, vec2_f32, vec3_f16, vec3_f32, vec4_f16, vec4_f32);
    _output.vec2_f16.x = _274.vec2_f16.y;
    _output.vec2_f16.y = _274.vec2_f16.x;
    scalar_f16_1 = _output.scalar_f16;
    scalar_f32_1 = _output.scalar_f32;
    vec2_f16_1 = _output.vec2_f16;
    vec2_f32_1 = _output.vec2_f32;
    vec3_f16_1 = _output.vec3_f16;
    vec3_f32_1 = _output.vec3_f32;
    vec4_f16_1 = _output.vec4_f16;
    vec4_f32_1 = _output.vec4_f32;
}


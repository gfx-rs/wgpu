//////////////////////////////////
// Entry point: "direct" (comp) //
//////////////////////////////////
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

layout(set = 0, binding = 0, std430) readonly buffer _13_12
{
    float _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float16_t _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    f16vec2 _m0[];
} _18;

void main()
{
    _15._m0[gl_GlobalInvocationID.x] = float16_t(_12._m0[gl_GlobalInvocationID.x]);
}


//////////////////////////////////////
// Entry point: "direct_vec" (comp) //
//////////////////////////////////////
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

layout(set = 0, binding = 0, std430) readonly buffer _13_12
{
    float _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float16_t _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    f16vec2 _m0[];
} _18;

void main()
{
    _18._m0[gl_GlobalInvocationID.x] = f16vec2(vec2(_12._m0[gl_GlobalInvocationID.x], _12._m0[gl_GlobalInvocationID.x + 1u]));
}


//////////////////////////////////
// Entry point: "reused" (comp) //
//////////////////////////////////
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

layout(set = 0, binding = 0, std430) readonly buffer _13_12
{
    float _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float16_t _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    f16vec2 _m0[];
} _18;

void main()
{
    float _68 = _12._m0[0u];
    float16_t _69 = float16_t(_68);
    _15._m0[0u] = _69;
    _15._m0[1u] = _69;
}


//////////////////////////////////////
// Entry point: "arithmetic" (comp) //
//////////////////////////////////////
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

layout(set = 0, binding = 0, std430) readonly buffer _13_12
{
    float _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float16_t _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    f16vec2 _m0[];
} _18;

void main()
{
    _15._m0[0u] = float16_t(_12._m0[0u]) + float16_t(1.0);
}


///////////////////////////////////////
// Entry point: "local_store" (comp) //
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
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) readonly buffer _13_12
{
    float _m0[];
} _12;

layout(set = 0, binding = 1, std430) buffer _16_15
{
    float16_t _m0[];
} _15;

layout(set = 0, binding = 2, std430) buffer _19_18
{
    f16vec2 _m0[];
} _18;

void main()
{
    float16_t _87 = float16_t(0.0);
    _87 = float16_t(_12._m0[0u]);
    _15._m0[0u] = _87;
}


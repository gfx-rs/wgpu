#version 460
#if defined(GL_ARB_gpu_shader_int64)
#extension GL_ARB_gpu_shader_int64 : require
#else
#error No extension available for 64-bit integers.
#endif
#extension GL_EXT_shader_image_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
layout(local_size_x = 2, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, r64ui) uniform u64image2D _9;

void main()
{
    uint64_t _25 = imageAtomicMax(_9, ivec2(0), 1ul);
    barrier();
    uint64_t _30 = imageAtomicMin(_9, ivec2(0), 1ul);
}


#version 460
#extension GL_EXT_spirv_intrinsics : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

spirv_instruction (extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4450)
int spvSDot_int_uint_uint(uint arg0, uint arg1, spirv_literal uint packedFormat);

spirv_instruction (extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4451)
uint spvUDot_uint_uint_uint(uint arg0, uint arg1, spirv_literal uint packedFormat);

vec2 _8()
{
    vec2 _15 = fma(vec2(2.0), vec2(0.5), vec2(0.5));
    return _15;
}

int _17()
{
    return 32;
}

uint _50()
{
    uint _61 = spvUDot_uint_uint_uint(3u, 4u, 0);
    return spvUDot_uint_uint_uint(7u + _61, 8u + _61, 0);
}

void main()
{
}


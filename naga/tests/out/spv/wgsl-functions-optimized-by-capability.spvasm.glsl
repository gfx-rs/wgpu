#version 460
#extension GL_EXT_spirv_intrinsics : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

spirv_instruction (extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4450)
int spvSDot_int_uint_uint(uint arg0, uint arg1, spirv_literal uint packedFormat);

spirv_instruction (extensions = ["SPV_KHR_integer_dot_product"], capabilities = [6019, 6018], id = 4451)
uint spvUDot_uint_uint_uint(uint arg0, uint arg1, spirv_literal uint packedFormat);

uint _5()
{
    uint _18 = spvUDot_uint_uint_uint(3u, 4u, 0);
    return spvUDot_uint_uint_uint(7u + _18, 8u + _18, 0);
}

void main()
{
}


#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
    vec4 _48 = vec4(0.0);
    uvec3 _42 = uvec3(0u);
    ivec4 _36 = ivec4(0);
    int _30 = 0;
    uvec4 _44 = uvec4(0u);
    uint _38 = 0u;
    ivec2 _32 = ivec2(0);
    vec2 _46 = vec2(0.0);
    uvec2 _40 = uvec2(0u);
    ivec3 _34 = ivec3(0);
    _38 = packSnorm4x8(_48);
    _38 = packUnorm4x8(_48);
    _38 = packSnorm2x16(_46);
    _38 = packUnorm2x16(_46);
    _38 = packHalf2x16(_46);
    _38 = pack32(u8vec4(_36));
    _38 = pack32(u8vec4(_44));
    _38 = pack32(u8vec4(clamp(_36, ivec4(-128), ivec4(127))));
    _38 = pack32(u8vec4(clamp(_44, uvec4(0u), uvec4(255u))));
    _48 = unpackSnorm4x8(_38);
    _48 = unpackUnorm4x8(_38);
    _46 = unpackSnorm2x16(_38);
    _46 = unpackUnorm2x16(_38);
    _46 = unpackHalf2x16(_38);
    _36 = ivec4(unpack8(_38));
    _44 = uvec4(unpack8(_38));
    uint _105 = min(5u, 32u);
    _30 = bitfieldInsert(_30, _30, int(_105), int(min(10u, (32u - _105))));
    uint _111 = min(5u, 32u);
    _32 = bitfieldInsert(_32, _32, int(_111), int(min(10u, (32u - _111))));
    uint _117 = min(5u, 32u);
    _34 = bitfieldInsert(_34, _34, int(_117), int(min(10u, (32u - _117))));
    uint _123 = min(5u, 32u);
    _36 = bitfieldInsert(_36, _36, int(_123), int(min(10u, (32u - _123))));
    uint _129 = min(5u, 32u);
    _38 = bitfieldInsert(_38, _38, int(_129), int(min(10u, (32u - _129))));
    uint _135 = min(5u, 32u);
    _40 = bitfieldInsert(_40, _40, int(_135), int(min(10u, (32u - _135))));
    uint _141 = min(5u, 32u);
    _42 = bitfieldInsert(_42, _42, int(_141), int(min(10u, (32u - _141))));
    uint _147 = min(5u, 32u);
    _44 = bitfieldInsert(_44, _44, int(_147), int(min(10u, (32u - _147))));
    uint _152 = min(5u, 32u);
    _30 = bitfieldExtract(_30, int(_152), int(min(10u, (32u - _152))));
    uint _157 = min(5u, 32u);
    _32 = bitfieldExtract(_32, int(_157), int(min(10u, (32u - _157))));
    uint _162 = min(5u, 32u);
    _34 = bitfieldExtract(_34, int(_162), int(min(10u, (32u - _162))));
    uint _167 = min(5u, 32u);
    _36 = bitfieldExtract(_36, int(_167), int(min(10u, (32u - _167))));
    uint _172 = min(5u, 32u);
    _38 = bitfieldExtract(_38, int(_172), int(min(10u, (32u - _172))));
    uint _177 = min(5u, 32u);
    _40 = bitfieldExtract(_40, int(_177), int(min(10u, (32u - _177))));
    uint _182 = min(5u, 32u);
    _42 = bitfieldExtract(_42, int(_182), int(min(10u, (32u - _182))));
    uint _187 = min(5u, 32u);
    _44 = bitfieldExtract(_44, int(_187), int(min(10u, (32u - _187))));
    _30 = findLSB(_30);
    _40 = uvec2(findLSB(_40));
    _34 = findMSB(_34);
    _42 = uvec3(findMSB(_42));
    _30 = findMSB(_30);
    _38 = uint(findMSB(_38));
    _30 = bitCount(_30);
    _32 = bitCount(_32);
    _34 = bitCount(_34);
    _36 = bitCount(_36);
    _38 = uint(bitCount(_38));
    _40 = uvec2(bitCount(_40));
    _42 = uvec3(bitCount(_42));
    _44 = uvec4(bitCount(_44));
    _30 = bitfieldReverse(_30);
    _32 = bitfieldReverse(_32);
    _34 = bitfieldReverse(_34);
    _36 = bitfieldReverse(_36);
    _38 = bitfieldReverse(_38);
    _40 = bitfieldReverse(_40);
    _42 = bitfieldReverse(_42);
    _44 = bitfieldReverse(_44);
}


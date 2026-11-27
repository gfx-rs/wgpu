#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) buffer _19_18
{
    vec3 _m0[];
} _18;

uvec4 _15[1] = uvec4[](uvec4(0u));
shared mat2x4 _21;

void _150(inout uvec4 _149)
{
    _149 = uvec4(_149.x, 1u, 2u, _149.w);
    uvec4 _157 = _149;
    _149 = uvec4(_149.x, 100u, 99u, _149.w);
    _149.y = 50u;
    _149 = uvec4(_157.wyz.xy.x, _149.yz, _157.wyz.xy.y);
}

uint _24()
{
    _15[0u] += uvec4(2u, 4u, 8u, 10u);
    return 0u;
}

uvec2 _39()
{
    _15[0u].x = 900u;
    _15[0u].z = 100u;
    return uvec2(55u, 66u);
}

uvec4 _51()
{
    _15[0u] = uvec4(0u);
    uint _56 = _24();
    uvec2 _57 = _39();
    _15[_56] = uvec4(_57.yx, _15[_56].zw);
    uvec4 _67 = _15[0u];
    _15[0u] = uvec4(0u);
    uint _69 = _24();
    uvec4 _71 = _15[_69];
    uvec2 _73 = _39();
    _15[_69] = uvec4((_71.yx + _73).yx, _15[_69].zw);
    return _67 + _15[0u];
}

uvec4 _88(int _87)
{
    uvec4 _100 = uvec4(0u);
    _100 = uvec4(_100.x, 1u, 2u, _100.w);
    uvec4 _107 = _100;
    _100 = uvec4(1u, 2u, 3u, 4u);
    _100 = uvec4(4u, 3u, 2u, 1u);
    _100 = uvec4(_100.x, 100u, 99u, _100.w);
    _100.z = 99u;
    _100.z++;
    _100.y = 50u;
    _100[(_87 == 1) ? 0 : 3] = 7u;
    _100[_87] = 8u;
    uvec2 _128 = _100.xz * uvec2(3u);
    _100 = uvec4(_128.x, _100.y, _128.y, _100.w);
    uvec2 _138 = _100.yw << uvec2(1u, 2u);
    _100 = uvec4(_100.x, _138.x, _100.z, _138.y);
    return _100 + _107.wyz.xyzx;
}

void main()
{
    uvec4 _186 = uvec4(0u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _21 = mat2x4(vec4(0.0), vec4(0.0));
    }
    barrier();
    _150(_186);
    _18._m0[gl_LocalInvocationIndex] = vec3(2.0, _18._m0[gl_LocalInvocationIndex].y, 1.0);
    _18._m0[gl_LocalInvocationIndex] = vec3(_18._m0[gl_LocalInvocationIndex].x, _18._m0[gl_LocalInvocationIndex].yz - vec2(1.0));
    _21[1u] = vec4(_21[1u].x, 4.0, _21[1u].z, 3.0);
    uvec4 _219 = _51();
    _18._m0[0u] = vec3(((_219 + _88(1)) + _186).xyz) + _21[1u].xyz;
}


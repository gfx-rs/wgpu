////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    uint _m0;
    uint _m1[4];
};

layout(set = 0, binding = 0, std430) buffer _22_21
{
    _6 _m0;
} _21;

layout(set = 0, binding = 1, std430) buffer _25_24
{
    uint _m0[];
} _24;

layout(set = 0, binding = 2, std430) readonly buffer _28_27
{
    float _m0[];
} _27;

layout(set = 0, binding = 3) uniform texture2D _30;
layout(set = 0, binding = 4) uniform sampler _32;

shared uint _34[64];
uint _36[4] = uint[](0u, 0u, 0u, 0u);

uint _41(uint _40)
{
    return _40;
}

void main()
{
    bool _88 = false;
    int _81 = 0;
    uint _74 = 0u;
    uint _70[2] = uint[](0u, 0u);
    uint _90 = 0u;
    bool _83 = false;
    uint _76 = 0u;
    uint _72 = 0u;
    uint _92 = 0u;
    uint _86 = 0u;
    uint _73 = 3u;
    uvec2 _129 = uvec2(4294967295u);
    uvec2 _180 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _34 = uint[](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    barrier();
    _21._m0._m0 = gl_LocalInvocationIndex;
    _24._m0[gl_LocalInvocationIndex] = 1u;
    uint _106 = _21._m0._m1[1u];
    _21._m0._m1[1u] = _106 + 2u;
    _74 = _106;
    uint _111 = _21._m0._m1[2u];
    _21._m0._m1[2u] = _111 + 3u;
    _76 = _111;
    _24._m0[0u] = _74 + _76;
    do
    {
        _83 = false;
        int _78 = 0;
        for (;;)
        {
            if (all(equal(uvec2(0u), _129)))
            {
                break;
            }
            _129 -= uvec2(uint(_129.y == 0u), 1u);
            if (!(_78 < 64))
            {
                break;
            }
            if (_34[_78] == gl_LocalInvocationIndex)
            {
                _81 = _78;
                _83 = true;
                break;
            }
            _78++;
            continue;
        }
        if (_83)
        {
            break;
        }
        _81 = -1;
        break;
    } while(false);
    _24._m0[1u] = uint(_81);
    do
    {
        _88 = false;
        switch (_24._m0[3u])
        {
            case 0u:
            {
                _86 = 10u;
                _88 = true;
                break;
            }
            case 1u:
            {
                _24._m0[3u] = 2u;
                break;
            }
            default:
            {
                _86 = 30u;
                _88 = true;
                break;
            }
        }
        if (_88)
        {
            break;
        }
        _86 = 20u;
        break;
    } while(false);
    _24._m0[2u] = _86;
    for (;;)
    {
        if (all(equal(uvec2(0u), _180)))
        {
            break;
        }
        _180 -= uvec2(uint(_180.y == 0u), 1u);
        if (!(_72 < 2u))
        {
            break;
        }
        _90 = 1u;
        _70[_72] += _90;
        _90++;
        _92 = 1u;
        _70[_72] += _92;
        _92++;
        _72++;
        continue;
    }
    _36[1u] = 7u;
    do
    {
        if (gl_LocalInvocationIndex == 0u)
        {
            break;
        }
        _24._m0[4u] = 5u;
        break;
    } while(false);
    _24._m0[5u] = (_70[0u] + _36[1u]) + _41(_73);
}


////////////////////////////////
// Entry point: "frag" (frag) //
////////////////////////////////
#version 460

struct _6
{
    uint _m0;
    uint _m1[4];
};

layout(set = 0, binding = 0, std430) buffer _22_21
{
    _6 _m0;
} _21;

layout(set = 0, binding = 1, std430) buffer _25_24
{
    uint _m0[];
} _24;

layout(set = 0, binding = 2, std430) readonly buffer _28_27
{
    float _m0[];
} _27;

layout(set = 0, binding = 3) uniform texture2D _30;
layout(set = 0, binding = 4) uniform sampler _32;

layout(location = 0) in vec2 _231;
layout(location = 0) out vec4 _234;
shared uint _34[64];
uint _36[4] = uint[](0u, 0u, 0u, 0u);

void main()
{
    float _244 = 0.0;
    bool _247 = false;
    uvec2 _256 = uvec2(4294967295u);
    do
    {
        _247 = false;
        uint _242 = 0u;
        for (;;)
        {
            if (all(equal(uvec2(0u), _256)))
            {
                break;
            }
            _256 -= uvec2(uint(_256.y == 0u), 1u);
            if (!(_242 < uint(_27._m0.length())))
            {
                break;
            }
            if (_27._m0[_242] > 0.0)
            {
                _244 = _27._m0[_242];
                _247 = true;
                break;
            }
            _242++;
            continue;
        }
        if (_247)
        {
            break;
        }
        _244 = 0.0;
        break;
    } while(false);
    _234 = texture(sampler2D(_30, _32), _231) * _244;
}


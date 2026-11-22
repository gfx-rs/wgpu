////////////////////////////////
// Entry point: "main" (comp) //
////////////////////////////////
#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

struct _6
{
    uint _m0;
    uint _m1[4];
};

layout(set = 0, binding = 0, std430) buffer _24_23
{
    _6 _m0;
} _23;

layout(set = 0, binding = 1, std430) buffer _27_26
{
    uint _m0[];
} _26;

layout(set = 0, binding = 2, std430) readonly buffer _30_29
{
    float _m0[];
} _29;

layout(set = 0, binding = 3) uniform texture2D _32;
layout(set = 0, binding = 4) uniform sampler _34;

shared uint _36[64];
uint _38[4] = uint[](0u, 0u, 0u, 0u);

uint _67(uint _66)
{
    return _66;
}

void main()
{
    uvec2 _136 = uvec2(0u);
    uint _129 = 0u;
    uint _123 = 0u;
    uint _109 = 0u;
    uint _103 = 0u;
    uint _99 = 0u;
    uvec2 _133 = uvec2(0u);
    uint _127 = 0u;
    bool _120 = false;
    uint _113 = 0u;
    uint _107 = 0u;
    uint _101 = 0u;
    uint _97[2] = uint[](0u, 0u);
    uint _138 = 0u;
    bool _131 = false;
    bool _125 = false;
    int _118 = 0;
    uint _111 = 0u;
    uint _105 = 0u;
    uint _100 = 3u;
    _138 = gl_LocalInvocationIndex;
    for (;;)
    {
        if (_138 >= 64u)
        {
            break;
        }
        _36[_138] = 0u;
        _138++;
        continue;
    }
    barrier();
    _101 = gl_LocalInvocationIndex;
    _23._m0._m0 = _101;
    _103 = 1u;
    _26._m0[gl_LocalInvocationIndex] = _103;
    _105 = 2u;
    uint _161 = _23._m0._m1[1u];
    _23._m0._m1[1u] = _161 + _105;
    _107 = _161;
    _109 = 3u;
    uint _167 = _23._m0._m1[2u];
    _23._m0._m1[2u] = _167 + _109;
    _111 = _167;
    _26._m0[0u] = _107 + _111;
    _113 = gl_LocalInvocationIndex;
    do
    {
        _120 = false;
        _133 = uvec2(4294967295u);
        int _115 = 0;
        for (;;)
        {
            if (all(equal(_133, uvec2(0u))))
            {
                break;
            }
            _133 -= uvec2(uint(_133.y == 0u), 1u);
            if (!(_115 < 64))
            {
                break;
            }
            if (_36[_115] == _113)
            {
                _118 = _115;
                _120 = true;
                break;
            }
            _115++;
            continue;
        }
        if (_120)
        {
            break;
        }
        _118 = -1;
        break;
    } while(false);
    _26._m0[1u] = uint(_118);
    do
    {
        _125 = false;
        switch (_26._m0[3u])
        {
            case 0u:
            {
                _123 = 10u;
                _125 = true;
                break;
            }
            case 1u:
            {
                _26._m0[3u] = 2u;
                break;
            }
            default:
            {
                _123 = 30u;
                _125 = true;
                break;
            }
        }
        if (_125)
        {
            break;
        }
        _123 = 20u;
        break;
    } while(false);
    _26._m0[2u] = _123;
    _136 = uvec2(4294967295u);
    for (;;)
    {
        if (all(equal(_136, uvec2(0u))))
        {
            break;
        }
        _136 -= uvec2(uint(_136.y == 0u), 1u);
        if (!(_99 < 2u))
        {
            break;
        }
        _127 = 1u;
        _97[_99] += _127;
        _127++;
        _129 = 1u;
        _97[_99] += _129;
        _129++;
        _99++;
        continue;
    }
    _38[1u] = 7u;
    _131 = gl_LocalInvocationIndex == 0u;
    do
    {
        if (_131)
        {
            break;
        }
        _26._m0[4u] = 5u;
        break;
    } while(false);
    _26._m0[5u] = (_97[0u] + _38[1u]) + _67(_100);
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

layout(set = 0, binding = 0, std430) buffer _24_23
{
    _6 _m0;
} _23;

layout(set = 0, binding = 1, std430) buffer _27_26
{
    uint _m0[];
} _26;

layout(set = 0, binding = 2, std430) readonly buffer _30_29
{
    float _m0[];
} _29;

layout(set = 0, binding = 3) uniform texture2D _32;
layout(set = 0, binding = 4) uniform sampler _34;

layout(location = 0) in vec2 _280;
layout(location = 0) out vec4 _283;
shared uint _36[64];
uint _38[4] = uint[](0u, 0u, 0u, 0u);

void main()
{
    float _293 = 0.0;
    uvec2 _298 = uvec2(0u);
    bool _296 = false;
    do
    {
        _296 = false;
        _298 = uvec2(4294967295u);
        uint _291 = 0u;
        for (;;)
        {
            if (all(equal(_298, uvec2(0u))))
            {
                break;
            }
            _298 -= uvec2(uint(_298.y == 0u), 1u);
            if (!(_291 < uint(_29._m0.length())))
            {
                break;
            }
            if (_29._m0[_291] > 0.0)
            {
                _293 = _29._m0[_291];
                _296 = true;
                break;
            }
            _291++;
            continue;
        }
        if (_296)
        {
            break;
        }
        _293 = 0.0;
        break;
    } while(false);
    _283 = texture(sampler2D(_32, _34), _280) * _293;
}


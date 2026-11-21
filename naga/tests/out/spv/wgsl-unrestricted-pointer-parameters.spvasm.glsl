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

layout(set = 0, binding = 0, std430) buffer _26_25
{
    _6 _m0;
} _25;

layout(set = 0, binding = 1, std430) buffer _29_28
{
    uint _m0[];
} _28;

layout(set = 0, binding = 2, std430) readonly buffer _32_31
{
    float _m0[];
} _31;

layout(set = 0, binding = 3) uniform texture2D _34;
layout(set = 0, binding = 4) uniform sampler _36;

shared uint _38[64];
uint _39[4] = uint[](0u, 0u, 0u, 0u);

uint _68(uint _67)
{
    return _67;
}

void main()
{
    uint _127 = 0u;
    uint _121 = 0u;
    uint _107 = 0u;
    uint _101 = 0u;
    uint _97 = 0u;
    uint _125 = 0u;
    bool _118 = false;
    uint _111 = 0u;
    uint _105 = 0u;
    uint _99 = 0u;
    uint _95[2] = uint[](0u, 0u);
    bool _129 = false;
    bool _123 = false;
    int _116 = 0;
    uint _109 = 0u;
    uint _103 = 0u;
    uint _98 = 3u;
    uvec2 _170 = uvec2(4294967295u);
    uvec2 _221 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _38 = uint[](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    barrier();
    _99 = gl_LocalInvocationIndex;
    _25._m0._m0 = _99;
    _101 = 1u;
    _28._m0[gl_LocalInvocationIndex] = _101;
    _103 = 2u;
    uint _145 = _25._m0._m1[1u];
    _25._m0._m1[1u] = _145 + _103;
    _105 = _145;
    _107 = 3u;
    uint _151 = _25._m0._m1[2u];
    _25._m0._m1[2u] = _151 + _107;
    _109 = _151;
    _28._m0[0u] = _105 + _109;
    _111 = gl_LocalInvocationIndex;
    do
    {
        _118 = false;
        int _113 = 0;
        for (;;)
        {
            if (all(equal(uvec2(0u), _170)))
            {
                break;
            }
            _170 -= uvec2(uint(_170.y == 0u), 1u);
            if (!(_113 < 64))
            {
                break;
            }
            if (_38[_113] == _111)
            {
                _116 = _113;
                _118 = true;
                break;
            }
            _113++;
            continue;
        }
        if (_118)
        {
            break;
        }
        _116 = -1;
        break;
    } while(false);
    _28._m0[1u] = uint(_116);
    do
    {
        _123 = false;
        switch (_28._m0[3u])
        {
            case 0u:
            {
                _121 = 10u;
                _123 = true;
                break;
            }
            case 1u:
            {
                _28._m0[3u] = 2u;
                break;
            }
            default:
            {
                _121 = 30u;
                _123 = true;
                break;
            }
        }
        if (_123)
        {
            break;
        }
        _121 = 20u;
        break;
    } while(false);
    _28._m0[2u] = _121;
    for (;;)
    {
        if (all(equal(uvec2(0u), _221)))
        {
            break;
        }
        _221 -= uvec2(uint(_221.y == 0u), 1u);
        if (!(_97 < 2u))
        {
            break;
        }
        _125 = 1u;
        _95[_97] += _125;
        _125++;
        _127 = 1u;
        _95[_97] += _127;
        _127++;
        _97++;
        continue;
    }
    _39[1u] = 7u;
    _129 = gl_LocalInvocationIndex == 0u;
    do
    {
        if (_129)
        {
            break;
        }
        _28._m0[4u] = 5u;
        break;
    } while(false);
    _28._m0[5u] = (_95[0u] + _39[1u]) + _68(_98);
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

layout(set = 0, binding = 0, std430) buffer _26_25
{
    _6 _m0;
} _25;

layout(set = 0, binding = 1, std430) buffer _29_28
{
    uint _m0[];
} _28;

layout(set = 0, binding = 2, std430) readonly buffer _32_31
{
    float _m0[];
} _31;

layout(set = 0, binding = 3) uniform texture2D _34;
layout(set = 0, binding = 4) uniform sampler _36;

layout(location = 0) in vec2 _272;
layout(location = 0) out vec4 _275;
shared uint _38[64];
uint _39[4] = uint[](0u, 0u, 0u, 0u);

void main()
{
    float _284 = 0.0;
    bool _287 = false;
    uvec2 _296 = uvec2(4294967295u);
    do
    {
        _287 = false;
        uint _282 = 0u;
        for (;;)
        {
            if (all(equal(uvec2(0u), _296)))
            {
                break;
            }
            _296 -= uvec2(uint(_296.y == 0u), 1u);
            if (!(_282 < uint(_31._m0.length())))
            {
                break;
            }
            if (_31._m0[_282] > 0.0)
            {
                _284 = _31._m0[_282];
                _287 = true;
                break;
            }
            _282++;
            continue;
        }
        if (_287)
        {
            break;
        }
        _284 = 0.0;
        break;
    } while(false);
    _275 = texture(sampler2D(_34, _36), _272) * _284;
}


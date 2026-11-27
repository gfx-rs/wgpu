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

layout(set = 0, binding = 0, std430) buffer _23_22
{
    _6 _m0;
} _22;

layout(set = 0, binding = 1, std430) buffer _26_25
{
    uint _m0[];
} _25;

layout(set = 0, binding = 2, std430) readonly buffer _29_28
{
    float _m0[];
} _28;

layout(set = 0, binding = 3) uniform texture2D _31;
layout(set = 0, binding = 4) uniform sampler _33;

shared uint _35[64];
uint _37[4] = uint[](0u, 0u, 0u, 0u);

uint _66(uint _65)
{
    return _65;
}

void main()
{
    uint _125 = 0u;
    uint _119 = 0u;
    uint _105 = 0u;
    uint _99 = 0u;
    uint _95 = 0u;
    uint _123 = 0u;
    bool _116 = false;
    uint _109 = 0u;
    uint _103 = 0u;
    uint _97 = 0u;
    uint _93[2] = uint[](0u, 0u);
    bool _127 = false;
    bool _121 = false;
    int _114 = 0;
    uint _107 = 0u;
    uint _101 = 0u;
    uint _96 = 3u;
    uvec2 _169 = uvec2(4294967295u);
    uvec2 _220 = uvec2(4294967295u);
    if (gl_LocalInvocationIndex == 0u)
    {
        _35 = uint[](0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
    }
    barrier();
    _97 = gl_LocalInvocationIndex;
    _22._m0._m0 = _97;
    _99 = 1u;
    _25._m0[gl_LocalInvocationIndex] = _99;
    _101 = 2u;
    uint _144 = _22._m0._m1[1u];
    _22._m0._m1[1u] = _144 + _101;
    _103 = _144;
    _105 = 3u;
    uint _150 = _22._m0._m1[2u];
    _22._m0._m1[2u] = _150 + _105;
    _107 = _150;
    _25._m0[0u] = _103 + _107;
    _109 = gl_LocalInvocationIndex;
    do
    {
        _116 = false;
        int _111 = 0;
        for (;;)
        {
            if (all(equal(uvec2(0u), _169)))
            {
                break;
            }
            _169 -= uvec2(uint(_169.y == 0u), 1u);
            if (!(_111 < 64))
            {
                break;
            }
            if (_35[_111] == _109)
            {
                _114 = _111;
                _116 = true;
                break;
            }
            _111++;
            continue;
        }
        if (_116)
        {
            break;
        }
        _114 = -1;
        break;
    } while(false);
    _25._m0[1u] = uint(_114);
    do
    {
        _121 = false;
        switch (_25._m0[3u])
        {
            case 0u:
            {
                _119 = 10u;
                _121 = true;
                break;
            }
            case 1u:
            {
                _25._m0[3u] = 2u;
                break;
            }
            default:
            {
                _119 = 30u;
                _121 = true;
                break;
            }
        }
        if (_121)
        {
            break;
        }
        _119 = 20u;
        break;
    } while(false);
    _25._m0[2u] = _119;
    for (;;)
    {
        if (all(equal(uvec2(0u), _220)))
        {
            break;
        }
        _220 -= uvec2(uint(_220.y == 0u), 1u);
        if (!(_95 < 2u))
        {
            break;
        }
        _123 = 1u;
        _93[_95] += _123;
        _123++;
        _125 = 1u;
        _93[_95] += _125;
        _125++;
        _95++;
        continue;
    }
    _37[1u] = 7u;
    _127 = gl_LocalInvocationIndex == 0u;
    do
    {
        if (_127)
        {
            break;
        }
        _25._m0[4u] = 5u;
        break;
    } while(false);
    _25._m0[5u] = (_93[0u] + _37[1u]) + _66(_96);
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

layout(set = 0, binding = 0, std430) buffer _23_22
{
    _6 _m0;
} _22;

layout(set = 0, binding = 1, std430) buffer _26_25
{
    uint _m0[];
} _25;

layout(set = 0, binding = 2, std430) readonly buffer _29_28
{
    float _m0[];
} _28;

layout(set = 0, binding = 3) uniform texture2D _31;
layout(set = 0, binding = 4) uniform sampler _33;

layout(location = 0) in vec2 _271;
layout(location = 0) out vec4 _274;
shared uint _35[64];
uint _37[4] = uint[](0u, 0u, 0u, 0u);

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
            if (!(_282 < uint(_28._m0.length())))
            {
                break;
            }
            if (_28._m0[_282] > 0.0)
            {
                _284 = _28._m0[_282];
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
    _274 = texture(sampler2D(_31, _33), _271) * _284;
}


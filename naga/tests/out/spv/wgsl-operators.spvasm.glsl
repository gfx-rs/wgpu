#version 460
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

vec4 _26()
{
    vec4 _36 = mix(vec4(0.0), vec4(1.0), vec4(0.5));
    return ((((vec4(ivec4(int(true)) + ivec4(0)) + vec4(bvec4(true))) + _36) + mix(vec4(0.0), vec4(1.0), vec4(0.100000001490116119384765625))) + vec4(intBitsToFloat(1))) + intBitsToFloat(ivec4(1));
}

ivec4 _50(ivec4 _52, ivec4 _53)
{
    bvec4 _56 = equal(_53, ivec4(0));
    bvec4 _61 = equal(_52, ivec4(int(0x80000000)));
    bvec4 _62 = equal(_53, ivec4(-1));
    bvec4 _63 = bvec4(_61.x && _62.x, _61.y && _62.y, _61.z && _62.z, _61.w && _62.w);
    ivec4 _66 = mix(_53, ivec4(1), bvec4(_56.x || _63.x, _56.y || _63.y, _56.z || _63.z, _56.w || _63.w));
    return _52 - ((_52 / _66) * _66);
}

vec4 _73(float _71, int _72)
{
    return (((vec2(2.0) + vec2(_71)) - vec2(4.0)) / vec2(8.0)).xyxy + vec4(_50(ivec4(_72), ivec4(2)));
}

vec2 _94()
{
    vec2 _99 = vec2(2.0);
    _99 += vec2(1.0);
    _99 -= vec2(3.0);
    _99 /= vec2(4.0);
    return _99;
}

vec3 _111(vec3 _110)
{
    return vec3(notEqual(_110, vec3(0.0)));
}

bool _120()
{
    return true;
}

bool _124()
{
    return false;
}

bool _127()
{
    return true;
}

bool _130()
{
    return false;
}

void _133()
{
    bool _150 = false;
    bool _144 = false;
    bool _154 = false;
    bool _148 = false;
    bool _141 = false;
    bool _152 = false;
    bool _146 = false;
    if (!true)
    {
        _141 = false;
    }
    else
    {
        _141 = true;
    }
    if (true)
    {
        _144 = false;
    }
    else
    {
        _144 = false;
    }
    if (!false)
    {
        _146 = false;
    }
    else
    {
        _146 = true;
    }
    if (!_120())
    {
        _148 = _124();
    }
    else
    {
        _148 = true;
    }
    if (_148)
    {
        if (!_127())
        {
            _152 = _130();
        }
        else
        {
            _152 = true;
        }
        _150 = _152;
    }
    else
    {
        _150 = false;
    }
    if (false)
    {
        _154 = _124();
    }
    else
    {
        _154 = true;
    }
}

int _201(int _203, int _204)
{
    return _203 / (((_204 == 0) || ((_203 == int(0x80000000)) && (_204 == (-1)))) ? 1 : _204);
}

uint _213(uint _215, uint _216)
{
    return _215 / ((_216 == 0u) ? 1u : _216);
}

ivec2 _224(ivec2 _226, ivec2 _227)
{
    bvec2 _230 = equal(_227, ivec2(0));
    bvec2 _233 = equal(_226, ivec2(int(0x80000000)));
    bvec2 _234 = equal(_227, ivec2(-1));
    bvec2 _235 = bvec2(_233.x && _234.x, _233.y && _234.y);
    return _226 / mix(_227, ivec2(1), bvec2(_230.x || _235.x, _230.y || _235.y));
}

uvec3 _240(uvec3 _242, uvec3 _243)
{
    return _242 / mix(_243, uvec3(1u), equal(_243, uvec3(0u)));
}

int _250(int _251, int _252)
{
    int _259 = ((_252 == 0) || ((_251 == int(0x80000000)) && (_252 == (-1)))) ? 1 : _252;
    return _251 - ((_251 / _259) * _259);
}

uint _263(uint _264, uint _265)
{
    return _264 % ((_265 == 0u) ? 1u : _265);
}

ivec2 _270(ivec2 _271, ivec2 _272)
{
    bvec2 _274 = equal(_272, ivec2(0));
    bvec2 _275 = equal(_271, ivec2(int(0x80000000)));
    bvec2 _276 = equal(_272, ivec2(-1));
    bvec2 _277 = bvec2(_275.x && _276.x, _275.y && _276.y);
    ivec2 _279 = mix(_272, ivec2(1), bvec2(_274.x || _277.x, _274.y || _277.y));
    return _271 - ((_271 / _279) * _279);
}

uvec3 _283(uvec3 _284, uvec3 _285)
{
    return _284 % mix(_285, uvec3(1u), equal(_285, uvec3(0u)));
}

uvec2 _291(uvec2 _293, uvec2 _294)
{
    return _293 / mix(_294, uvec2(1u), equal(_294, uvec2(0u)));
}

uvec2 _301(uvec2 _302, uvec2 _303)
{
    return _302 % mix(_303, uvec2(1u), equal(_303, uvec2(0u)));
}

void _309()
{
    int _321 = 0;
    int _324 = 0;
    float _327 = -1.0;
    ivec2 _328 = -ivec2(1);
    vec2 _329 = -vec2(1.0);
    int _330 = 2 + 1;
    uint _331 = 2u + 1u;
    float _332 = 2.0 + 1.0;
    ivec2 _333 = ivec2(2) + ivec2(1);
    uvec3 _334 = uvec3(2u) + uvec3(1u);
    vec4 _335 = vec4(2.0) + vec4(1.0);
    int _336 = 2 - 1;
    uint _337 = 2u - 1u;
    float _338 = 2.0 - 1.0;
    ivec2 _339 = ivec2(2) - ivec2(1);
    uvec3 _340 = uvec3(2u) - uvec3(1u);
    vec4 _341 = vec4(2.0) - vec4(1.0);
    int _342 = 2 * 1;
    uint _343 = 2u * 1u;
    float _344 = 2.0 * 1.0;
    ivec2 _345 = ivec2(2) * ivec2(1);
    uvec3 _346 = uvec3(2u) * uvec3(1u);
    vec4 _347 = vec4(2.0) * vec4(1.0);
    float _350 = 2.0 / 1.0;
    vec4 _353 = vec4(2.0) / vec4(1.0);
    float _356 = 2.0 - 1.0 * trunc(2.0 / 1.0);
    vec4 _359 = vec4(2.0) - vec4(1.0) * trunc(vec4(2.0) / vec4(1.0));
    ivec2 _362 = ivec2(2) + ivec2(1);
    ivec2 _363 = ivec2(2) + ivec2(1);
    uvec2 _364 = uvec2(2u) + uvec2(1u);
    uvec2 _365 = uvec2(2u) + uvec2(1u);
    vec2 _366 = vec2(2.0) + vec2(1.0);
    vec2 _367 = vec2(2.0) + vec2(1.0);
    ivec2 _368 = ivec2(2) - ivec2(1);
    ivec2 _369 = ivec2(2) - ivec2(1);
    uvec2 _370 = uvec2(2u) - uvec2(1u);
    uvec2 _371 = uvec2(2u) - uvec2(1u);
    vec2 _372 = vec2(2.0) - vec2(1.0);
    vec2 _373 = vec2(2.0) - vec2(1.0);
    vec2 _382 = vec2(2.0) * 1.0;
    vec2 _383 = vec2(1.0) * 2.0;
    vec2 _388 = vec2(2.0) / vec2(1.0);
    vec2 _389 = vec2(2.0) / vec2(1.0);
    vec2 _394 = vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0));
    vec2 _395 = vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0));
    mat3 _396 = mat3(vec3(0.0), vec3(0.0), vec3(0.0)) * 1.0;
    mat3 _397 = mat3(vec3(0.0), vec3(0.0), vec3(0.0)) * 2.0;
    vec3 _398 = mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)) * vec4(1.0);
    vec4 _399 = vec3(2.0) * mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    _324 = _321 + int(0x80000000);
}

void _403()
{
}

void _430()
{
}

void _469()
{
    int _471 = 0;
    ivec3 _473 = ivec3(0);
    _471 = 1;
    _471++;
    _471--;
    _471 *= _471;
    _471 = _201(_471, _471);
    _471 = _250(_471, 1);
    _471 &= 0;
    _471 |= 0;
    _471 ^= 0;
    _471 = _471 << int(2u);
    _471 = _471 >> int(1u);
    _471++;
    _471--;
    _473.y++;
    _473.y--;
}

void _511()
{
    int _513 = -1;
    int _514 = -1;
    int _516 = -1;
    int _518 = -1;
    int _520 = -1;
    int _523 = -1;
    int _527 = -1;
    int _532 = -1;
    float _537 = -1.0;
    float _538 = -1.0;
    float _540 = -1.0;
    float _542 = -1.0;
    float _544 = -1.0;
    float _547 = -1.0;
    float _551 = -1.0;
    float _556 = -1.0;
}

void main()
{
    _133();
    _309();
    _403();
    _430();
    _469();
    _511();
}


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
    return _52 - mix(_53, ivec4(1), bvec4(_56.x || _63.x, _56.y || _63.y, _56.z || _63.z, _56.w || _63.w)) * (_52 / mix(_53, ivec4(1), bvec4(_56.x || _63.x, _56.y || _63.y, _56.z || _63.z, _56.w || _63.w)));
}

vec4 _71(float _69, int _70)
{
    return (((vec2(2.0) + vec2(_69)) - vec2(4.0)) / vec2(8.0)).xyxy + vec4(_50(ivec4(_70), ivec4(2)));
}

vec2 _92()
{
    vec2 _97 = vec2(2.0);
    _97 += vec2(1.0);
    _97 -= vec2(3.0);
    _97 /= vec2(4.0);
    return _97;
}

vec3 _109(vec3 _108)
{
    return vec3(notEqual(_108, vec3(0.0)));
}

bool _118()
{
    return true;
}

bool _122()
{
    return false;
}

bool _125()
{
    return true;
}

bool _128()
{
    return false;
}

void _131()
{
    bool _148 = false;
    bool _142 = false;
    bool _152 = false;
    bool _146 = false;
    bool _139 = false;
    bool _150 = false;
    bool _144 = false;
    if (!true)
    {
        _139 = false;
    }
    else
    {
        _139 = true;
    }
    if (true)
    {
        _142 = false;
    }
    else
    {
        _142 = false;
    }
    if (!false)
    {
        _144 = false;
    }
    else
    {
        _144 = true;
    }
    if (!_118())
    {
        _146 = _122();
    }
    else
    {
        _146 = true;
    }
    if (_146)
    {
        if (!_125())
        {
            _150 = _128();
        }
        else
        {
            _150 = true;
        }
        _148 = _150;
    }
    else
    {
        _148 = false;
    }
    if (false)
    {
        _152 = _122();
    }
    else
    {
        _152 = true;
    }
}

int _199(int _201, int _202)
{
    return _201 / (((_202 == 0) || ((_201 == int(0x80000000)) && (_202 == (-1)))) ? 1 : _202);
}

uint _211(uint _213, uint _214)
{
    return _213 / ((_214 == 0u) ? 1u : _214);
}

ivec2 _222(ivec2 _224, ivec2 _225)
{
    bvec2 _228 = equal(_225, ivec2(0));
    bvec2 _231 = equal(_224, ivec2(int(0x80000000)));
    bvec2 _232 = equal(_225, ivec2(-1));
    bvec2 _233 = bvec2(_231.x && _232.x, _231.y && _232.y);
    return _224 / mix(_225, ivec2(1), bvec2(_228.x || _233.x, _228.y || _233.y));
}

uvec3 _238(uvec3 _240, uvec3 _241)
{
    return _240 / mix(_241, uvec3(1u), equal(_241, uvec3(0u)));
}

int _248(int _249, int _250)
{
    return _249 - (((_250 == 0) || ((_249 == int(0x80000000)) && (_250 == (-1)))) ? 1 : _250) * (_249 / (((_250 == 0) || ((_249 == int(0x80000000)) && (_250 == (-1)))) ? 1 : _250));
}

uint _259(uint _260, uint _261)
{
    return _260 % ((_261 == 0u) ? 1u : _261);
}

ivec2 _266(ivec2 _267, ivec2 _268)
{
    bvec2 _270 = equal(_268, ivec2(0));
    bvec2 _271 = equal(_267, ivec2(int(0x80000000)));
    bvec2 _272 = equal(_268, ivec2(-1));
    bvec2 _273 = bvec2(_271.x && _272.x, _271.y && _272.y);
    return _267 - mix(_268, ivec2(1), bvec2(_270.x || _273.x, _270.y || _273.y)) * (_267 / mix(_268, ivec2(1), bvec2(_270.x || _273.x, _270.y || _273.y)));
}

uvec3 _277(uvec3 _278, uvec3 _279)
{
    return _278 % mix(_279, uvec3(1u), equal(_279, uvec3(0u)));
}

uvec2 _285(uvec2 _287, uvec2 _288)
{
    return _287 / mix(_288, uvec2(1u), equal(_288, uvec2(0u)));
}

uvec2 _295(uvec2 _296, uvec2 _297)
{
    return _296 % mix(_297, uvec2(1u), equal(_297, uvec2(0u)));
}

void _303()
{
    int _315 = 0;
    int _318 = 0;
    float _321 = -1.0;
    ivec2 _322 = -ivec2(1);
    vec2 _323 = -vec2(1.0);
    int _324 = 2 + 1;
    uint _325 = 2u + 1u;
    float _326 = 2.0 + 1.0;
    ivec2 _327 = ivec2(2) + ivec2(1);
    uvec3 _328 = uvec3(2u) + uvec3(1u);
    vec4 _329 = vec4(2.0) + vec4(1.0);
    int _330 = 2 - 1;
    uint _331 = 2u - 1u;
    float _332 = 2.0 - 1.0;
    ivec2 _333 = ivec2(2) - ivec2(1);
    uvec3 _334 = uvec3(2u) - uvec3(1u);
    vec4 _335 = vec4(2.0) - vec4(1.0);
    int _336 = 2 * 1;
    uint _337 = 2u * 1u;
    float _338 = 2.0 * 1.0;
    ivec2 _339 = ivec2(2) * ivec2(1);
    uvec3 _340 = uvec3(2u) * uvec3(1u);
    vec4 _341 = vec4(2.0) * vec4(1.0);
    float _344 = 2.0 / 1.0;
    vec4 _347 = vec4(2.0) / vec4(1.0);
    float _350 = 2.0 - 1.0 * trunc(2.0 / 1.0);
    vec4 _353 = vec4(2.0) - vec4(1.0) * trunc(vec4(2.0) / vec4(1.0));
    ivec2 _356 = ivec2(2) + ivec2(1);
    ivec2 _357 = ivec2(2) + ivec2(1);
    uvec2 _358 = uvec2(2u) + uvec2(1u);
    uvec2 _359 = uvec2(2u) + uvec2(1u);
    vec2 _360 = vec2(2.0) + vec2(1.0);
    vec2 _361 = vec2(2.0) + vec2(1.0);
    ivec2 _362 = ivec2(2) - ivec2(1);
    ivec2 _363 = ivec2(2) - ivec2(1);
    uvec2 _364 = uvec2(2u) - uvec2(1u);
    uvec2 _365 = uvec2(2u) - uvec2(1u);
    vec2 _366 = vec2(2.0) - vec2(1.0);
    vec2 _367 = vec2(2.0) - vec2(1.0);
    vec2 _376 = vec2(2.0) * 1.0;
    vec2 _377 = vec2(1.0) * 2.0;
    vec2 _382 = vec2(2.0) / vec2(1.0);
    vec2 _383 = vec2(2.0) / vec2(1.0);
    vec2 _388 = vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0));
    vec2 _389 = vec2(2.0) - vec2(1.0) * trunc(vec2(2.0) / vec2(1.0));
    mat3 _390 = mat3(vec3(0.0), vec3(0.0), vec3(0.0)) * 1.0;
    mat3 _391 = mat3(vec3(0.0), vec3(0.0), vec3(0.0)) * 2.0;
    vec3 _392 = mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0)) * vec4(1.0);
    vec4 _393 = vec3(2.0) * mat4x3(vec3(0.0), vec3(0.0), vec3(0.0), vec3(0.0));
    _318 = _315 + int(0x80000000);
}

void _397()
{
}

void _424()
{
}

void _463()
{
    int _465 = 0;
    ivec3 _467 = ivec3(0);
    _465 = 1;
    _465++;
    _465--;
    _465 *= _465;
    _465 = _199(_465, _465);
    _465 = _248(_465, 1);
    _465 &= 0;
    _465 |= 0;
    _465 ^= 0;
    _465 = _465 << int(2u);
    _465 = _465 >> int(1u);
    _465++;
    _465--;
    _467.y++;
    _467.y--;
}

void _505()
{
    int _507 = -1;
    int _508 = -1;
    int _510 = -1;
    int _512 = -1;
    int _514 = -1;
    int _517 = -1;
    int _521 = -1;
    int _526 = -1;
    float _531 = -1.0;
    float _532 = -1.0;
    float _534 = -1.0;
    float _536 = -1.0;
    float _538 = -1.0;
    float _541 = -1.0;
    float _545 = -1.0;
    float _550 = -1.0;
}

void main()
{
    _131();
    _303();
    _397();
    _424();
    _463();
    _505();
}


#version 460

struct _11
{
    float _m0;
    float _m1;
};

struct _12
{
    vec2 _m0;
    vec2 _m1;
};

struct _13
{
    vec4 _m0;
    vec4 _m1;
};

struct _14
{
    float _m0;
    int _m1;
};

struct _15
{
    vec4 _m0;
    ivec4 _m1;
};

void main()
{
    float _53 = 1.0;
    int _55 = 1;
    float _58 = degrees(1.0);
    float _59 = radians(1.0);
    vec4 _60 = degrees(vec4(0.0));
    vec4 _61 = radians(vec4(0.0));
    vec4 _62 = clamp(vec4(0.0), vec4(0.0), vec4(1.0));
    vec4 _64 = refract(vec4(0.0), vec4(0.0), 1.0);
    float _72 = ldexp(1.0, 2);
    vec2 _73 = ldexp(vec2(1.0, 2.0), ivec2(3, 4));
    _11 _74;
    _74._m0 = modf(1.5, _74._m1);
    _11 _75;
    _75._m0 = modf(1.5, _75._m1);
    _11 _77;
    _77._m0 = modf(1.5, _77._m1);
    _12 _79;
    _79._m0 = modf(vec2(1.5), _79._m1);
    _13 _80;
    _80._m0 = modf(vec4(1.5), _80._m1);
    _12 _83;
    _83._m0 = modf(vec2(1.5), _83._m1);
    _14 _86;
    _86._m0 = frexp(1.5, _86._m1);
    _14 _87;
    _87._m0 = frexp(1.5, _87._m1);
    _14 _89;
    _89._m0 = frexp(1.5, _89._m1);
    _15 _91;
    _91._m0 = frexp(vec4(1.5), _91._m1);
}


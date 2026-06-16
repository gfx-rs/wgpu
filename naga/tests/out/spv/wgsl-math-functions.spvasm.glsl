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
    float _53 = degrees(1.0);
    float _54 = radians(1.0);
    vec4 _55 = degrees(vec4(0.0));
    vec4 _56 = radians(vec4(0.0));
    vec4 _57 = clamp(vec4(0.0), vec4(0.0), vec4(1.0));
    vec4 _59 = refract(vec4(0.0), vec4(0.0), 1.0);
    float _60 = ldexp(1.0, 2);
    vec2 _61 = ldexp(vec2(1.0, 2.0), ivec2(3, 4));
    _11 _62;
    _62._m0 = modf(1.5, _62._m1);
    _11 _63;
    _63._m0 = modf(1.5, _63._m1);
    _11 _65;
    _65._m0 = modf(1.5, _65._m1);
    _12 _67;
    _67._m0 = modf(vec2(1.5), _67._m1);
    _13 _68;
    _68._m0 = modf(vec4(1.5), _68._m1);
    _12 _71;
    _71._m0 = modf(vec2(1.5), _71._m1);
    _14 _74;
    _74._m0 = frexp(1.5, _74._m1);
    _14 _75;
    _75._m0 = frexp(1.5, _75._m1);
    _14 _77;
    _77._m0 = frexp(1.5, _77._m1);
    _15 _79;
    _79._m0 = frexp(vec4(1.5), _79._m1);
}


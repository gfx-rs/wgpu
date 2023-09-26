struct _modf_result_f32_ {
    float fract;
    float whole;
};

struct _modf_result_vec2_f32_ {
    float2 fract;
    float2 whole;
};

struct _modf_result_vec4_f32_ {
    float4 fract;
    float4 whole;
};

struct _frexp_result_f32_ {
    float fract;
    int exp_;
};

struct _frexp_result_vec4_f32_ {
    float4 fract;
    int4 exp_;
};

_modf_result_f32_ naga_modf(float arg) {
    float other;
    _modf_result_f32_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

_modf_result_vec2_f32_ naga_modf(float2 arg) {
    float2 other;
    _modf_result_vec2_f32_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

_modf_result_vec4_f32_ naga_modf(float4 arg) {
    float4 other;
    _modf_result_vec4_f32_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

_frexp_result_f32_ naga_frexp(float arg) {
    float other;
    _frexp_result_f32_ result;
    result.fract = sign(arg) * frexp(arg, other);
    result.exp_ = other;
    return result;
}

_frexp_result_vec4_f32_ naga_frexp(float4 arg) {
    float4 other;
    _frexp_result_vec4_f32_ result;
    result.fract = sign(arg) * frexp(arg, other);
    result.exp_ = other;
    return result;
}

void main()
{
    float4 v = (0.0).xxxx;
    float a = degrees(1.0);
    float b = radians(1.0);
    float4 c = degrees(v);
    float4 d = radians(v);
    float4 e = saturate(v);
    float4 g = refract(v, v, 1.0);
    int sign_a = sign(-1);
    int4 sign_b = sign((-1).xxxx);
    float sign_c = sign(-1.0);
    float4 sign_d = sign((-1.0).xxxx);
    int const_dot = dot((int2)0, (int2)0);
    uint first_leading_bit_abs = firstbithigh(abs(0u));
    int flb_a = asint(firstbithigh(-1));
    int2 flb_b = asint(firstbithigh((-1).xx));
    uint2 flb_c = firstbithigh((1u).xx);
    int ftb_a = asint(firstbitlow(-1));
    uint ftb_b = firstbitlow(1u);
    int2 ftb_c = asint(firstbitlow((-1).xx));
    uint2 ftb_d = firstbitlow((1u).xx);
    uint ctz_a = min(32u, firstbitlow(0u));
    int ctz_b = asint(min(32u, firstbitlow(0)));
    uint ctz_c = min(32u, firstbitlow(4294967295u));
    int ctz_d = asint(min(32u, firstbitlow(-1)));
    uint2 ctz_e = min((32u).xx, firstbitlow((0u).xx));
    int2 ctz_f = asint(min((32u).xx, firstbitlow((0).xx)));
    uint2 ctz_g = min((32u).xx, firstbitlow((1u).xx));
    int2 ctz_h = asint(min((32u).xx, firstbitlow((1).xx)));
    int clz_a = (-1 < 0 ? 0 : 31 - asint(firstbithigh(-1)));
    uint clz_b = (31u - firstbithigh(1u));
    int2 _expr68 = (-1).xx;
    int2 clz_c = (_expr68 < (0).xx ? (0).xx : (31).xx - asint(firstbithigh(_expr68)));
    uint2 clz_d = ((31u).xx - firstbithigh((1u).xx));
    float lde_a = ldexp(1.0, 2);
    float2 lde_b = ldexp(float2(1.0, 2.0), int2(3, 4));
    _modf_result_f32_ modf_a = naga_modf(1.5);
    float modf_b = naga_modf(1.5).fract;
    float modf_c = naga_modf(1.5).whole;
    _modf_result_vec2_f32_ modf_d = naga_modf(float2(1.5, 1.5));
    float modf_e = naga_modf(float4(1.5, 1.5, 1.5, 1.5)).whole.x;
    float modf_f = naga_modf(float2(1.5, 1.5)).fract.y;
    _frexp_result_f32_ frexp_a = naga_frexp(1.5);
    float frexp_b = naga_frexp(1.5).fract;
    int frexp_c = naga_frexp(1.5).exp_;
    int frexp_d = naga_frexp(float4(1.5, 1.5, 1.5, 1.5)).exp_.x;
}

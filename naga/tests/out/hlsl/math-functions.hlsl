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

int2 ZeroValueint2() {
    return (int2)0;
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
    int4 sign_b = int4(-1, -1, -1, -1);
    float4 sign_d = float4(-1.0, -1.0, -1.0, -1.0);
    int const_dot = dot(ZeroValueint2(), ZeroValueint2());
    int2 flb_b = int2(-1, -1);
    uint2 flb_c = uint2(0u, 0u);
    int2 ftb_c = int2(0, 0);
    uint2 ftb_d = uint2(0u, 0u);
    uint2 ctz_e = uint2(32u, 32u);
    int2 ctz_f = int2(32, 32);
    uint2 ctz_g = uint2(0u, 0u);
    int2 ctz_h = int2(0, 0);
    int2 clz_c = int2(0, 0);
    uint2 clz_d = uint2(31u, 31u);
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
    float quantizeToF16_a = f16tof32(f32tof16(1.0));
    float2 quantizeToF16_b = f16tof32(f32tof16(float2(1.0, 1.0)));
    float3 quantizeToF16_c = f16tof32(f32tof16(float3(1.0, 1.0, 1.0)));
    float4 quantizeToF16_d = f16tof32(f32tof16(float4(1.0, 1.0, 1.0, 1.0)));
}

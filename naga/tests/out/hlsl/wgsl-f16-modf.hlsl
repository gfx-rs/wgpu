struct _modf_result_f16_ {
    half fract;
    half whole;
};

struct _modf_result_vec2_f16_ {
    half2 fract;
    half2 whole;
};

struct _modf_result_vec3_f16_ {
    half3 fract;
    half3 whole;
};

struct _modf_result_vec4_f16_ {
    half4 fract;
    half4 whole;
};

_modf_result_f16_ naga_modf(float arg) {
    float other;
    _modf_result_f16_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

_modf_result_vec2_f16_ naga_modf(float2 arg) {
    float2 other;
    _modf_result_vec2_f16_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

_modf_result_vec3_f16_ naga_modf(float3 arg) {
    float3 other;
    _modf_result_vec3_f16_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

_modf_result_vec4_f16_ naga_modf(float4 arg) {
    float4 other;
    _modf_result_vec4_f16_ result;
    result.fract = modf(arg, other);
    result.whole = other;
    return result;
}

RWByteAddressBuffer output : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    half x = output.Load<half>(0);
    output.Store(0, naga_modf(x).fract);
    output.Store(2, naga_modf((x).xx).whole.x);
    output.Store(4, naga_modf((x).xxx).fract.y);
    output.Store(6, naga_modf((x).xxxx).whole.z);
    return;
}

// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct type_1 {
    half inner[4];
};
struct _modf_result_f16_ {
    half fract;
    half whole;
};
struct _modf_result_vec2_f16_ {
    metal::half2 fract;
    metal::half2 whole;
};
struct _modf_result_vec3_f16_ {
    metal::packed_half3 fract;
    metal::packed_half3 whole;
};
struct _modf_result_vec4_f16_ {
    metal::half4 fract;
    metal::half4 whole;
};

_modf_result_f16_ naga_modf(half arg) {
    half other;
    half fract = metal::modf(arg, other);
    return _modf_result_f16_{ fract, other };
}

_modf_result_vec2_f16_ naga_modf(metal::half2 arg) {
    metal::half2 other;
    metal::half2 fract = metal::modf(arg, other);
    return _modf_result_vec2_f16_{ fract, other };
}

_modf_result_vec3_f16_ naga_modf(metal::half3 arg) {
    metal::half3 other;
    metal::half3 fract = metal::modf(arg, other);
    return _modf_result_vec3_f16_{ fract, other };
}

_modf_result_vec4_f16_ naga_modf(metal::half4 arg) {
    metal::half4 other;
    metal::half4 fract = metal::modf(arg, other);
    return _modf_result_vec4_f16_{ fract, other };
}

kernel void main_(
  device type_1& output [[user(fake0)]]
) {
    half x = output.inner[0];
    output.inner[0] = naga_modf(x).fract;
    output.inner[1] = naga_modf(metal::half2(x)).whole.x;
    output.inner[2] = naga_modf(metal::half3(x)).fract[1];
    output.inner[3] = naga_modf(metal::half4(x)).whole.z;
    return;
}

// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

constant half MIN_F16_ = -65504.0h;
constant half MAX_F16_ = 65504.0h;
constant float MIN_F32_ = -340282350000000000000000000000000000000.0;
constant float MAX_F32_ = 340282350000000000000000000000000000000.0;

void test_const_eval(
) {
    int min_f16_to_i32_ = -65504;
    int max_f16_to_i32_ = 65504;
    uint min_f16_to_u32_ = 0u;
    uint max_f16_to_u32_ = 65504u;
    long min_f16_to_i64_ = -65504L;
    long max_f16_to_i64_ = 65504L;
    ulong min_f16_to_u64_ = 0uL;
    ulong max_f16_to_u64_ = 65504uL;
    int min_f32_to_i32_ = (-2147483647 - 1);
    int max_f32_to_i32_ = 2147483520;
    uint min_f32_to_u32_ = 0u;
    uint max_f32_to_u32_ = 4294967040u;
    long min_f32_to_i64_ = (-9223372036854775807L - 1L);
    long max_f32_to_i64_ = 9223371487098961920L;
    ulong min_f32_to_u64_ = 0uL;
    ulong max_f32_to_u64_ = 18446742974197923840uL;
    int min_abstract_float_to_i32_ = (-2147483647 - 1);
    int max_abstract_float_to_i32_ = 2147483647;
    uint min_abstract_float_to_u32_ = 0u;
    uint max_abstract_float_to_u32_ = 4294967295u;
    long min_abstract_float_to_i64_ = (-9223372036854775807L - 1L);
    long max_abstract_float_to_i64_ = 9223372036854774784L;
    ulong min_abstract_float_to_u64_ = 0uL;
    ulong max_abstract_float_to_u64_ = 18446744073709549568uL;
    return;
}

int naga_f2i32(half value) {
    return static_cast<int>(metal::clamp(value, -65504.0h, 65504.0h));
}

int test_f16_to_i32_(
    half f
) {
    return naga_f2i32(f);
}

uint naga_f2u32(half value) {
    return static_cast<uint>(metal::clamp(value, 0.0h, 65504.0h));
}

uint test_f16_to_u32_(
    half f_1
) {
    return naga_f2u32(f_1);
}

long naga_f2i64(half value) {
    return static_cast<long>(metal::clamp(value, -65504.0h, 65504.0h));
}

long test_f16_to_i64_(
    half f_2
) {
    return naga_f2i64(f_2);
}

ulong naga_f2u64(half value) {
    return static_cast<ulong>(metal::clamp(value, 0.0h, 65504.0h));
}

ulong test_f16_to_u64_(
    half f_3
) {
    return naga_f2u64(f_3);
}

int naga_f2i32(float value) {
    return static_cast<int>(metal::clamp(value, -2147483600.0, 2147483500.0));
}

int test_f32_to_i32_(
    float f_4
) {
    return naga_f2i32(f_4);
}

uint naga_f2u32(float value) {
    return static_cast<uint>(metal::clamp(value, 0.0, 4294967000.0));
}

uint test_f32_to_u32_(
    float f_5
) {
    return naga_f2u32(f_5);
}

long naga_f2i64(float value) {
    return static_cast<long>(metal::clamp(value, -9223372000000000000.0, 9223371500000000000.0));
}

long test_f32_to_i64_(
    float f_6
) {
    return naga_f2i64(f_6);
}

ulong naga_f2u64(float value) {
    return static_cast<ulong>(metal::clamp(value, 0.0, 18446743000000000000.0));
}

ulong test_f32_to_u64_(
    float f_7
) {
    return naga_f2u64(f_7);
}

metal::int2 naga_f2i32(metal::half2 value) {
    return static_cast<metal::int2>(metal::clamp(value, -65504.0h, 65504.0h));
}

metal::int2 test_f16_to_i32_vec(
    metal::half2 f_8
) {
    return naga_f2i32(f_8);
}

metal::uint2 naga_f2u32(metal::half2 value) {
    return static_cast<metal::uint2>(metal::clamp(value, 0.0h, 65504.0h));
}

metal::uint2 test_f16_to_u32_vec(
    metal::half2 f_9
) {
    return naga_f2u32(f_9);
}

metal::long2 naga_f2i64(metal::half2 value) {
    return static_cast<metal::long2>(metal::clamp(value, -65504.0h, 65504.0h));
}

metal::long2 test_f16_to_i64_vec(
    metal::half2 f_10
) {
    return naga_f2i64(f_10);
}

metal::ulong2 naga_f2u64(metal::half2 value) {
    return static_cast<metal::ulong2>(metal::clamp(value, 0.0h, 65504.0h));
}

metal::ulong2 test_f16_to_u64_vec(
    metal::half2 f_11
) {
    return naga_f2u64(f_11);
}

metal::int2 naga_f2i32(metal::float2 value) {
    return static_cast<metal::int2>(metal::clamp(value, -2147483600.0, 2147483500.0));
}

metal::int2 test_f32_to_i32_vec(
    metal::float2 f_12
) {
    return naga_f2i32(f_12);
}

metal::uint2 naga_f2u32(metal::float2 value) {
    return static_cast<metal::uint2>(metal::clamp(value, 0.0, 4294967000.0));
}

metal::uint2 test_f32_to_u32_vec(
    metal::float2 f_13
) {
    return naga_f2u32(f_13);
}

metal::long2 naga_f2i64(metal::float2 value) {
    return static_cast<metal::long2>(metal::clamp(value, -9223372000000000000.0, 9223371500000000000.0));
}

metal::long2 test_f32_to_i64_vec(
    metal::float2 f_14
) {
    return naga_f2i64(f_14);
}

metal::ulong2 naga_f2u64(metal::float2 value) {
    return static_cast<metal::ulong2>(metal::clamp(value, 0.0, 18446743000000000000.0));
}

metal::ulong2 test_f32_to_u64_vec(
    metal::float2 f_15
) {
    return naga_f2u64(f_15);
}

kernel void main_(
) {
    test_const_eval();
    int _e1 = test_f16_to_i32_(1.0h);
    uint _e3 = test_f16_to_u32_(1.0h);
    long _e5 = test_f16_to_i64_(1.0h);
    ulong _e7 = test_f16_to_u64_(1.0h);
    int _e9 = test_f32_to_i32_(1.0);
    uint _e11 = test_f32_to_u32_(1.0);
    long _e13 = test_f32_to_i64_(1.0);
    ulong _e15 = test_f32_to_u64_(1.0);
    metal::int2 _e19 = test_f16_to_i32_vec(metal::half2(1.0h, 2.0h));
    metal::uint2 _e23 = test_f16_to_u32_vec(metal::half2(1.0h, 2.0h));
    metal::long2 _e27 = test_f16_to_i64_vec(metal::half2(1.0h, 2.0h));
    metal::ulong2 _e31 = test_f16_to_u64_vec(metal::half2(1.0h, 2.0h));
    metal::int2 _e35 = test_f32_to_i32_vec(metal::float2(1.0, 2.0));
    metal::uint2 _e39 = test_f32_to_u32_vec(metal::float2(1.0, 2.0));
    metal::long2 _e43 = test_f32_to_i64_vec(metal::float2(1.0, 2.0));
    metal::ulong2 _e47 = test_f32_to_u64_vec(metal::float2(1.0, 2.0));
    return;
}

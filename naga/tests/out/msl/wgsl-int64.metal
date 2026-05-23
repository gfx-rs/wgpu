// language: metal2.3
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct UniformCompatible {
    uint val_u32_;
    int val_i32_;
    float val_f32_;
    char _pad3[4];
    ulong val_u64_;
    char _pad4[8];
    metal::ulong2 val_u64_2_;
    char _pad5[16];
    metal::ulong3 val_u64_3_;
    metal::ulong4 val_u64_4_;
    long val_i64_;
    char _pad8[8];
    metal::long2 val_i64_2_;
    metal::long3 val_i64_3_;
    metal::long4 val_i64_4_;
    ulong final_value;
    char _pad12[24];
};
struct type_11 {
    ulong inner[2];
};
struct type_12 {
    long inner[2];
};
struct StorageCompatible {
    type_11 val_u64_array_2_;
    type_12 val_i64_array_2_;
};
constant ulong constant_variable = 20uL;

long naga_f2i64(float value) {
    return static_cast<long>(metal::clamp(value, -9223372000000000000.0, 9223371500000000000.0));
}

long naga_abs(long val) {
    return metal::select(as_type<long>(-as_type<ulong>(val)), val, val >= 0);
}

long naga_dot_long2(metal::long2 a, metal::long2 b) {
    return ( + a.x * b.x + a.y * b.y);
}

long int64_function(
    long x,
    thread long& private_variable,
    constant UniformCompatible& input_uniform,
    device UniformCompatible const& input_storage,
    device StorageCompatible const& input_arrays,
    device UniformCompatible& output,
    device StorageCompatible& output_arrays
) {
    long val = 20L;
    long phony = private_variable;
    long _e5 = val;
    val = as_type<long>(as_type<ulong>(_e5) + as_type<ulong>(as_type<long>(as_type<ulong>(as_type<long>(as_type<ulong>(31L) - as_type<ulong>(1002003004005006L))) + as_type<ulong>(-9223372036854775807L))));
    long _e12 = val;
    long _e13 = val;
    val = as_type<long>(as_type<ulong>(_e12) + as_type<ulong>(as_type<long>(as_type<ulong>(_e13) + as_type<ulong>(5L))));
    long _e17 = val;
    uint _e20 = input_uniform.val_u32_;
    long _e21 = val;
    val = as_type<long>(as_type<ulong>(_e17) + as_type<ulong>(static_cast<long>(_e20 + static_cast<uint>(_e21))));
    long _e26 = val;
    int _e29 = input_uniform.val_i32_;
    long _e30 = val;
    val = as_type<long>(as_type<ulong>(_e26) + as_type<ulong>(static_cast<long>(as_type<int>(as_type<uint>(_e29) + as_type<uint>(static_cast<int>(_e30))))));
    long _e35 = val;
    float _e38 = input_uniform.val_f32_;
    long _e39 = val;
    val = as_type<long>(as_type<ulong>(_e35) + as_type<ulong>(naga_f2i64(_e38 + static_cast<float>(_e39))));
    long _e44 = val;
    long _e47 = input_uniform.val_i64_;
    val = as_type<long>(as_type<ulong>(_e44) + as_type<ulong>(metal::long3(_e47).z));
    long _e51 = val;
    ulong _e54 = input_uniform.val_u64_;
    val = as_type<long>(as_type<ulong>(_e51) + as_type<ulong>(as_type<long>(_e54)));
    long _e57 = val;
    metal::ulong2 _e60 = input_uniform.val_u64_2_;
    val = as_type<long>(as_type<ulong>(_e57) + as_type<ulong>(as_type<metal::long2>(_e60).y));
    long _e64 = val;
    metal::ulong3 _e67 = input_uniform.val_u64_3_;
    val = as_type<long>(as_type<ulong>(_e64) + as_type<ulong>(as_type<metal::long3>(_e67).z));
    long _e71 = val;
    metal::ulong4 _e74 = input_uniform.val_u64_4_;
    val = as_type<long>(as_type<ulong>(_e71) + as_type<ulong>(as_type<metal::long4>(_e74).w));
    long _e78 = val;
    val = as_type<long>(as_type<ulong>(_e78) + as_type<ulong>((-9223372036854775807L - 1L)));
    long _e85 = input_uniform.val_i64_;
    long _e88 = input_storage.val_i64_;
    output.val_i64_ = as_type<long>(as_type<ulong>(_e85) + as_type<ulong>(_e88));
    metal::long2 _e94 = input_uniform.val_i64_2_;
    metal::long2 _e97 = input_storage.val_i64_2_;
    output.val_i64_2_ = as_type<metal::long2>(as_type<metal::ulong2>(_e94) + as_type<metal::ulong2>(_e97));
    metal::long3 _e103 = input_uniform.val_i64_3_;
    metal::long3 _e106 = input_storage.val_i64_3_;
    output.val_i64_3_ = as_type<metal::long3>(as_type<metal::ulong3>(_e103) + as_type<metal::ulong3>(_e106));
    metal::long4 _e112 = input_uniform.val_i64_4_;
    metal::long4 _e115 = input_storage.val_i64_4_;
    output.val_i64_4_ = as_type<metal::long4>(as_type<metal::ulong4>(_e112) + as_type<metal::ulong4>(_e115));
    type_12 _e121 = input_arrays.val_i64_array_2_;
    output_arrays.val_i64_array_2_ = _e121;
    long _e122 = val;
    long _e123 = val;
    val = as_type<long>(as_type<ulong>(_e122) + as_type<ulong>(naga_abs(_e123)));
    long _e126 = val;
    long _e127 = val;
    long _e128 = val;
    long _e129 = val;
    val = as_type<long>(as_type<ulong>(_e126) + as_type<ulong>(metal::clamp(_e127, _e128, _e129)));
    long _e132 = val;
    long _e133 = val;
    long _e135 = val;
    val = as_type<long>(as_type<ulong>(_e132) + as_type<ulong>(naga_dot_long2(metal::long2(_e133), metal::long2(_e135))));
    long _e139 = val;
    long _e140 = val;
    long _e141 = val;
    val = as_type<long>(as_type<ulong>(_e139) + as_type<ulong>(metal::max(_e140, _e141)));
    long _e144 = val;
    long _e145 = val;
    long _e146 = val;
    val = as_type<long>(as_type<ulong>(_e144) + as_type<ulong>(metal::min(_e145, _e146)));
    long _e149 = val;
    long _e150 = val;
    val = as_type<long>(as_type<ulong>(_e149) + as_type<ulong>(metal::select(metal::select(long(-1), long(1), (_e150 > 0)), long(0), (_e150 == 0))));
    long _e153 = val;
    return _e153;
}

ulong naga_f2u64(float value) {
    return static_cast<ulong>(metal::clamp(value, 0.0, 18446743000000000000.0));
}

ulong naga_dot_ulong2(metal::ulong2 a, metal::ulong2 b) {
    return ( + a.x * b.x + a.y * b.y);
}

ulong uint64_function(
    ulong x_1,
    constant UniformCompatible& input_uniform,
    device UniformCompatible const& input_storage,
    device StorageCompatible const& input_arrays,
    device UniformCompatible& output,
    device StorageCompatible& output_arrays
) {
    ulong val_1 = 20uL;
    ulong _e3 = val_1;
    val_1 = _e3 + ((31uL + 18446744073709551615uL) - 18446744073709551615uL);
    ulong _e10 = val_1;
    ulong _e11 = val_1;
    val_1 = _e10 + (_e11 + 5uL);
    ulong _e15 = val_1;
    uint _e18 = input_uniform.val_u32_;
    ulong _e19 = val_1;
    val_1 = _e15 + static_cast<ulong>(_e18 + static_cast<uint>(_e19));
    ulong _e24 = val_1;
    int _e27 = input_uniform.val_i32_;
    ulong _e28 = val_1;
    val_1 = _e24 + static_cast<ulong>(as_type<int>(as_type<uint>(_e27) + as_type<uint>(static_cast<int>(_e28))));
    ulong _e33 = val_1;
    float _e36 = input_uniform.val_f32_;
    ulong _e37 = val_1;
    val_1 = _e33 + naga_f2u64(_e36 + static_cast<float>(_e37));
    ulong _e42 = val_1;
    ulong _e45 = input_uniform.val_u64_;
    val_1 = _e42 + metal::ulong3(_e45).z;
    ulong _e49 = val_1;
    long _e52 = input_uniform.val_i64_;
    val_1 = _e49 + as_type<ulong>(_e52);
    ulong _e55 = val_1;
    metal::long2 _e58 = input_uniform.val_i64_2_;
    val_1 = _e55 + as_type<metal::ulong2>(_e58).y;
    ulong _e62 = val_1;
    metal::long3 _e65 = input_uniform.val_i64_3_;
    val_1 = _e62 + as_type<metal::ulong3>(_e65).z;
    ulong _e69 = val_1;
    metal::long4 _e72 = input_uniform.val_i64_4_;
    val_1 = _e69 + as_type<metal::ulong4>(_e72).w;
    ulong _e80 = input_uniform.val_u64_;
    ulong _e83 = input_storage.val_u64_;
    output.val_u64_ = _e80 + _e83;
    metal::ulong2 _e89 = input_uniform.val_u64_2_;
    metal::ulong2 _e92 = input_storage.val_u64_2_;
    output.val_u64_2_ = _e89 + _e92;
    metal::ulong3 _e98 = input_uniform.val_u64_3_;
    metal::ulong3 _e101 = input_storage.val_u64_3_;
    output.val_u64_3_ = _e98 + _e101;
    metal::ulong4 _e107 = input_uniform.val_u64_4_;
    metal::ulong4 _e110 = input_storage.val_u64_4_;
    output.val_u64_4_ = _e107 + _e110;
    type_11 _e116 = input_arrays.val_u64_array_2_;
    output_arrays.val_u64_array_2_ = _e116;
    ulong _e117 = val_1;
    ulong _e118 = val_1;
    val_1 = _e117 + metal::abs(_e118);
    ulong _e121 = val_1;
    ulong _e122 = val_1;
    ulong _e123 = val_1;
    ulong _e124 = val_1;
    val_1 = _e121 + metal::clamp(_e122, _e123, _e124);
    ulong _e127 = val_1;
    ulong _e128 = val_1;
    ulong _e130 = val_1;
    val_1 = _e127 + naga_dot_ulong2(metal::ulong2(_e128), metal::ulong2(_e130));
    ulong _e134 = val_1;
    ulong _e135 = val_1;
    ulong _e136 = val_1;
    val_1 = _e134 + metal::max(_e135, _e136);
    ulong _e139 = val_1;
    ulong _e140 = val_1;
    ulong _e141 = val_1;
    val_1 = _e139 + metal::min(_e140, _e141);
    ulong _e144 = val_1;
    return _e144;
}

[[max_total_threads_per_threadgroup(1)]] kernel void main_(
  constant UniformCompatible& input_uniform [[user(fake0)]]
, device UniformCompatible const& input_storage [[user(fake0)]]
, device StorageCompatible const& input_arrays [[user(fake0)]]
, device UniformCompatible& output [[user(fake0)]]
, device StorageCompatible& output_arrays [[user(fake0)]]
) {
    long private_variable = 1L;
    ulong _e3 = uint64_function(67uL, input_uniform, input_storage, input_arrays, output, output_arrays);
    long _e5 = int64_function(60L, private_variable, input_uniform, input_storage, input_arrays, output, output_arrays);
    output.final_value = _e3 + as_type<ulong>(_e5);
    return;
}

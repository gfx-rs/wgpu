// language: metal2.4
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct UniformCompatible {
    uint val_u32_;
    int val_i32_;
    float val_f32_;
    ushort val_u16_;
    char _pad4[2];
    metal::ushort2 val_u16_2_;
    char _pad5[4];
    metal::ushort3 val_u16_3_;
    metal::ushort4 val_u16_4_;
    short val_i16_;
    char _pad8[2];
    metal::short2 val_i16_2_;
    metal::short3 val_i16_3_;
    metal::short4 val_i16_4_;
    ushort final_value;
    char _pad12[6];
};
struct type_11 {
    ushort inner[2];
};
struct type_12 {
    short inner[2];
};
struct StorageCompatible {
    type_11 val_u16_array_2_;
    type_12 val_i16_array_2_;
};
struct type_13 {
    short inner[4];
};
constant ushort constant_variable = static_cast<ushort>(20);
constant short f16_to_i16_clamped = static_cast<short>(32767);

short naga_abs(short val) {
    return metal::select(as_type<short>(static_cast<ushort>(-as_type<ushort>(val))), val, val >= short(0));
}

short naga_div(short lhs, short rhs) {
    return lhs / metal::select(rhs, short(1), (lhs == static_cast<short>(-32768) & rhs == short(-1)) | (rhs == short(0)));
}

short naga_mod(short lhs, short rhs) {
    short divisor = metal::select(rhs, short(1), (lhs == static_cast<short>(-32768) & rhs == short(-1)) | (rhs == short(0)));
    return lhs - (lhs / divisor) * divisor;
}

short naga_neg(short val) {
    return as_type<short>(static_cast<ushort>(-as_type<ushort>(val)));
}

short int16_function(
    short x,
    thread short& private_variable,
    constant UniformCompatible& input_uniform,
    device UniformCompatible const& input_storage,
    device StorageCompatible const& input_arrays,
    device UniformCompatible& output,
    device StorageCompatible& output_arrays
) {
    short val = static_cast<short>(20);
    type_13 arr = type_13 {{static_cast<short>(1), static_cast<short>(2), static_cast<short>(3), static_cast<short>(4)}};
    short phony = private_variable;
    short _e5 = val;
    val = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e5)) + as_type<ushort>(static_cast<ushort>(static_cast<short>(5)))));
    short _e8 = val;
    uint _e11 = input_uniform.val_u32_;
    val = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e8)) + as_type<ushort>(static_cast<ushort>(static_cast<short>(_e11)))));
    short _e14 = val;
    int _e17 = input_uniform.val_i32_;
    val = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e14)) + as_type<ushort>(static_cast<ushort>(static_cast<short>(_e17)))));
    short _e20 = val;
    short _e23 = input_uniform.val_i16_;
    val = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e20)) + as_type<ushort>(static_cast<ushort>(metal::short3(_e23).z))));
    short _e31 = input_uniform.val_i16_;
    short _e34 = input_storage.val_i16_;
    output.val_i16_ = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e31)) + as_type<ushort>(static_cast<ushort>(_e34))));
    metal::short2 _e40 = input_uniform.val_i16_2_;
    metal::short2 _e43 = input_storage.val_i16_2_;
    output.val_i16_2_ = as_type<metal::short2>(static_cast<metal::ushort2>(as_type<metal::ushort2>(static_cast<metal::ushort2>(_e40)) + as_type<metal::ushort2>(static_cast<metal::ushort2>(_e43))));
    metal::short3 _e49 = input_uniform.val_i16_3_;
    metal::short3 _e52 = input_storage.val_i16_3_;
    output.val_i16_3_ = as_type<metal::short3>(static_cast<metal::ushort3>(as_type<metal::ushort3>(static_cast<metal::ushort3>(_e49)) + as_type<metal::ushort3>(static_cast<metal::ushort3>(_e52))));
    metal::short4 _e58 = input_uniform.val_i16_4_;
    metal::short4 _e61 = input_storage.val_i16_4_;
    output.val_i16_4_ = as_type<metal::short4>(static_cast<metal::ushort4>(as_type<metal::ushort4>(static_cast<metal::ushort4>(_e58)) + as_type<metal::ushort4>(static_cast<metal::ushort4>(_e61))));
    type_12 _e67 = input_arrays.val_i16_array_2_;
    output_arrays.val_i16_array_2_ = _e67;
    short _e68 = val;
    val = naga_abs(_e68);
    short _e70 = val;
    short _e71 = val;
    val = metal::max(_e70, _e71);
    short _e73 = val;
    short _e74 = val;
    val = metal::min(_e73, _e74);
    short _e76 = val;
    short _e77 = val;
    short _e78 = val;
    val = metal::clamp(_e76, _e77, _e78);
    short _e80 = val;
    val = metal::select(metal::select(short(-1), short(1), (_e80 > 0)), short(0), (_e80 == 0));
    short _e82 = val;
    val = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e82)) - as_type<ushort>(static_cast<ushort>(static_cast<short>(1)))));
    short _e85 = val;
    val = as_type<short>(static_cast<ushort>(as_type<ushort>(static_cast<ushort>(_e85)) * as_type<ushort>(static_cast<ushort>(static_cast<short>(2)))));
    short _e88 = val;
    val = naga_div(_e88, static_cast<short>(3));
    short _e91 = val;
    val = naga_mod(_e91, static_cast<short>(4));
    short _e94 = val;
    val = _e94 & static_cast<short>(255);
    short _e97 = val;
    val = _e97 | static_cast<short>(16);
    short _e100 = val;
    val = _e100 ^ static_cast<short>(1);
    short _e103 = val;
    val = _e103 << 2u;
    short _e106 = val;
    val = _e106 >> 1u;
    short _e109 = val;
    val = naga_neg(_e109);
    short _e111 = val;
    bool cmp_lt = _e111 < static_cast<short>(0);
    short _e114 = val;
    bool cmp_le = _e114 <= static_cast<short>(0);
    short _e117 = val;
    bool cmp_gt = _e117 > static_cast<short>(0);
    short _e120 = val;
    bool cmp_ge = _e120 >= static_cast<short>(0);
    short _e123 = val;
    bool cmp_eq = _e123 == static_cast<short>(0);
    short _e126 = val;
    bool cmp_ne = _e126 != static_cast<short>(0);
    val = cmp_lt ? static_cast<short>(2) : static_cast<short>(1);
    short _e139 = val;
    arr.inner[0] = _e139;
    short _e141 = arr.inner[1];
    val = _e141;
    short _e144 = arr.inner[static_cast<ushort>(1)];
    val = _e144;
    short _e147 = val;
    output.val_u32_ = static_cast<uint>(_e147);
    short _e151 = val;
    output.val_i32_ = static_cast<int>(_e151);
    short _e155 = val;
    output.val_f32_ = static_cast<float>(_e155);
    uint _e159 = output.val_u32_;
    val = static_cast<short>(_e159);
    short _e161 = val;
    ushort as_unsigned = as_type<ushort>(_e161);
    val = as_type<short>(as_unsigned);
    metal::short2 _e166 = input_uniform.val_i16_2_;
    metal::short2 _e169 = input_uniform.val_i16_2_;
    metal::short2 v = as_type<metal::short2>(static_cast<metal::ushort2>(as_type<metal::ushort2>(static_cast<metal::ushort2>(_e166)) + as_type<metal::ushort2>(static_cast<metal::ushort2>(_e169))));
    metal::short2 v2_ = as_type<metal::short2>(static_cast<metal::ushort2>(as_type<metal::ushort2>(static_cast<metal::ushort2>(v)) * as_type<metal::ushort2>(static_cast<metal::ushort2>(metal::short2(static_cast<short>(2))))));
    output.val_i16_2_ = v2_;
    short _e176 = val;
    return _e176;
}

ushort naga_div(ushort lhs, ushort rhs) {
    return lhs / metal::select(rhs, ushort(1), rhs == ushort(0));
}

ushort naga_mod(ushort lhs, ushort rhs) {
    return lhs % metal::select(rhs, ushort(1), rhs == ushort(0));
}

ushort uint16_function(
    ushort x_1,
    constant UniformCompatible& input_uniform,
    device UniformCompatible const& input_storage,
    device StorageCompatible const& input_arrays,
    device UniformCompatible& output,
    device StorageCompatible& output_arrays
) {
    ushort val_1 = static_cast<ushort>(20);
    ushort _e3 = val_1;
    val_1 = _e3 + static_cast<ushort>(5);
    ushort _e6 = val_1;
    uint _e9 = input_uniform.val_u32_;
    val_1 = _e6 + static_cast<ushort>(_e9);
    ushort _e12 = val_1;
    int _e15 = input_uniform.val_i32_;
    val_1 = _e12 + static_cast<ushort>(_e15);
    ushort _e18 = val_1;
    ushort _e21 = input_uniform.val_u16_;
    val_1 = _e18 + metal::ushort3(_e21).z;
    ushort _e29 = input_uniform.val_u16_;
    ushort _e32 = input_storage.val_u16_;
    output.val_u16_ = _e29 + _e32;
    metal::ushort2 _e38 = input_uniform.val_u16_2_;
    metal::ushort2 _e41 = input_storage.val_u16_2_;
    output.val_u16_2_ = _e38 + _e41;
    metal::ushort3 _e47 = input_uniform.val_u16_3_;
    metal::ushort3 _e50 = input_storage.val_u16_3_;
    output.val_u16_3_ = _e47 + _e50;
    metal::ushort4 _e56 = input_uniform.val_u16_4_;
    metal::ushort4 _e59 = input_storage.val_u16_4_;
    output.val_u16_4_ = _e56 + _e59;
    type_11 _e65 = input_arrays.val_u16_array_2_;
    output_arrays.val_u16_array_2_ = _e65;
    ushort _e66 = val_1;
    val_1 = metal::abs(_e66);
    ushort _e68 = val_1;
    ushort _e69 = val_1;
    val_1 = metal::max(_e68, _e69);
    ushort _e71 = val_1;
    ushort _e72 = val_1;
    val_1 = metal::min(_e71, _e72);
    ushort _e74 = val_1;
    ushort _e75 = val_1;
    ushort _e76 = val_1;
    val_1 = metal::clamp(_e74, _e75, _e76);
    ushort _e78 = val_1;
    val_1 = _e78 - static_cast<ushort>(1);
    ushort _e81 = val_1;
    val_1 = _e81 * static_cast<ushort>(2);
    ushort _e84 = val_1;
    val_1 = naga_div(_e84, static_cast<ushort>(3));
    ushort _e87 = val_1;
    val_1 = naga_mod(_e87, static_cast<ushort>(4));
    ushort _e90 = val_1;
    val_1 = _e90 & static_cast<ushort>(255);
    ushort _e93 = val_1;
    val_1 = _e93 | static_cast<ushort>(16);
    ushort _e96 = val_1;
    val_1 = _e96 ^ static_cast<ushort>(1);
    ushort _e101 = val_1;
    output.val_u32_ = static_cast<uint>(_e101);
    ushort _e105 = val_1;
    output.val_i32_ = static_cast<int>(_e105);
    ushort _e109 = val_1;
    output.val_f32_ = static_cast<float>(_e109);
    uint _e113 = output.val_u32_;
    val_1 = static_cast<ushort>(_e113);
    ushort _e115 = val_1;
    return _e115;
}

struct main_Input {
};
[[max_total_threads_per_threadgroup(64)]] kernel void main_(
  uint subgroup_invocation_id [[thread_index_in_simdgroup]]
, uint __local_invocation_index [[thread_index_in_threadgroup]]
, constant UniformCompatible& input_uniform [[user(fake0)]]
, device UniformCompatible const& input_storage [[user(fake0)]]
, device StorageCompatible const& input_arrays [[user(fake0)]]
, device UniformCompatible& output [[user(fake0)]]
, device StorageCompatible& output_arrays [[user(fake0)]]
, threadgroup ushort& shared_val
) {
    short private_variable = static_cast<short>(1);
    if (__local_invocation_index == 0u) {
        shared_val = {};
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    short sg_val = {};
    ushort sg_uval = {};
    shared_val = static_cast<ushort>(0);
    ushort _e6 = uint16_function(static_cast<ushort>(67), input_uniform, input_storage, input_arrays, output, output_arrays);
    short _e8 = int16_function(static_cast<short>(60), private_variable, input_uniform, input_storage, input_arrays, output, output_arrays);
    output.final_value = _e6 + static_cast<ushort>(_e8);
    sg_val = static_cast<short>(subgroup_invocation_id);
    short _e13 = sg_val;
    short unnamed = metal::simd_sum(_e13);
    sg_val = unnamed;
    short _e15 = sg_val;
    short unnamed_1 = metal::simd_product(_e15);
    sg_val = unnamed_1;
    short _e17 = sg_val;
    short unnamed_2 = metal::simd_min(_e17);
    sg_val = unnamed_2;
    short _e19 = sg_val;
    short unnamed_3 = metal::simd_max(_e19);
    sg_val = unnamed_3;
    short _e21 = sg_val;
    short unnamed_4 = metal::simd_prefix_exclusive_sum(_e21);
    sg_val = unnamed_4;
    short _e23 = sg_val;
    short unnamed_5 = metal::simd_prefix_inclusive_sum(_e23);
    sg_val = unnamed_5;
    short _e25 = sg_val;
    short unnamed_6 = metal::simd_broadcast_first(_e25);
    sg_val = unnamed_6;
    short _e27 = sg_val;
    short unnamed_7 = metal::simd_broadcast(_e27, 4u);
    sg_val = unnamed_7;
    sg_uval = static_cast<ushort>(subgroup_invocation_id);
    ushort _e32 = sg_uval;
    ushort unnamed_8 = metal::simd_sum(_e32);
    sg_uval = unnamed_8;
    ushort _e34 = sg_uval;
    ushort unnamed_9 = metal::simd_min(_e34);
    sg_uval = unnamed_9;
    ushort _e36 = sg_uval;
    ushort unnamed_10 = metal::simd_max(_e36);
    sg_uval = unnamed_10;
    short _e40 = sg_val;
    output.val_i16_ = _e40;
    ushort _e43 = sg_uval;
    output.val_u16_ = _e43;
    return;
}

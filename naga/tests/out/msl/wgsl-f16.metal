// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct UniformCompatible {
    uint val_u32_;
    int val_i32_;
    float val_f32_;
    half val_f16_;
    char _pad4[2];
    metal::half2 val_f16_2_;
    char _pad5[4];
    metal::half3 val_f16_3_;
    metal::half4 val_f16_4_;
    half final_value;
    char _pad8[2];
    metal::half2x2 val_mat2x2_;
    char _pad9[4];
    metal::half2x3 val_mat2x3_;
    metal::half2x4 val_mat2x4_;
    metal::half3x2 val_mat3x2_;
    char _pad12[4];
    metal::half3x3 val_mat3x3_;
    metal::half3x4 val_mat3x4_;
    metal::half4x2 val_mat4x2_;
    metal::half4x3 val_mat4x3_;
    metal::half4x4 val_mat4x4_;
};
struct type_16 {
    half inner[2];
};
struct StorageCompatible {
    type_16 val_f16_array_2_;
};
struct LayoutTest {
    half scalar1_;
    half scalar2_;
    char _pad2[4];
    metal::packed_half3 v3_;
    half tuck_in;
    half scalar4_;
    char _pad5[2];
    uint larger;
};
constant half constant_variable = 15.203125h;

half f16_function(
    half x,
    thread half& private_variable,
    constant UniformCompatible& input_uniform,
    device UniformCompatible const& input_storage,
    device StorageCompatible const& input_arrays,
    device UniformCompatible& output,
    device StorageCompatible& output_arrays
) {
    LayoutTest l = {};
    half val = 15.203125h;
    half phony = private_variable;
    half _e6 = val;
    val = _e6 + -33344.0h;
    half _e8 = val;
    half _e11 = val;
    val = _e11 + (_e8 + 5.0h);
    float _e15 = input_uniform.val_f32_;
    half _e16 = val;
    half _e20 = val;
    val = _e20 + static_cast<half>(_e15 + static_cast<float>(_e16));
    half _e24 = input_uniform.val_f16_;
    half _e27 = val;
    val = _e27 + metal::half3(_e24).z;
    output.val_i32_ = 65504;
    output.val_i32_ = -65504;
    output.val_u32_ = 65504u;
    output.val_u32_ = 0u;
    output.val_f32_ = 65504.0;
    output.val_f32_ = -65504.0;
    half _e51 = input_uniform.val_f16_;
    half _e54 = input_storage.val_f16_;
    output.val_f16_ = _e51 + _e54;
    metal::half2 _e60 = input_uniform.val_f16_2_;
    metal::half2 _e63 = input_storage.val_f16_2_;
    output.val_f16_2_ = _e60 + _e63;
    metal::half3 _e69 = input_uniform.val_f16_3_;
    metal::half3 _e72 = input_storage.val_f16_3_;
    output.val_f16_3_ = _e69 + _e72;
    metal::half4 _e78 = input_uniform.val_f16_4_;
    metal::half4 _e81 = input_storage.val_f16_4_;
    output.val_f16_4_ = _e78 + _e81;
    metal::half2x2 _e87 = input_uniform.val_mat2x2_;
    metal::half2x2 _e90 = input_storage.val_mat2x2_;
    output.val_mat2x2_ = _e87 + _e90;
    metal::half2x3 _e96 = input_uniform.val_mat2x3_;
    metal::half2x3 _e99 = input_storage.val_mat2x3_;
    output.val_mat2x3_ = _e96 + _e99;
    metal::half2x4 _e105 = input_uniform.val_mat2x4_;
    metal::half2x4 _e108 = input_storage.val_mat2x4_;
    output.val_mat2x4_ = _e105 + _e108;
    metal::half3x2 _e114 = input_uniform.val_mat3x2_;
    metal::half3x2 _e117 = input_storage.val_mat3x2_;
    output.val_mat3x2_ = _e114 + _e117;
    metal::half3x3 _e123 = input_uniform.val_mat3x3_;
    metal::half3x3 _e126 = input_storage.val_mat3x3_;
    output.val_mat3x3_ = _e123 + _e126;
    metal::half3x4 _e132 = input_uniform.val_mat3x4_;
    metal::half3x4 _e135 = input_storage.val_mat3x4_;
    output.val_mat3x4_ = _e132 + _e135;
    metal::half4x2 _e141 = input_uniform.val_mat4x2_;
    metal::half4x2 _e144 = input_storage.val_mat4x2_;
    output.val_mat4x2_ = _e141 + _e144;
    metal::half4x3 _e150 = input_uniform.val_mat4x3_;
    metal::half4x3 _e153 = input_storage.val_mat4x3_;
    output.val_mat4x3_ = _e150 + _e153;
    metal::half4x4 _e159 = input_uniform.val_mat4x4_;
    metal::half4x4 _e162 = input_storage.val_mat4x4_;
    output.val_mat4x4_ = _e159 + _e162;
    type_16 _e168 = input_arrays.val_f16_array_2_;
    output_arrays.val_f16_array_2_ = _e168;
    half _e169 = val;
    half _e171 = val;
    val = _e171 + metal::abs(_e169);
    half _e173 = val;
    half _e174 = val;
    half _e175 = val;
    half _e177 = val;
    val = _e177 + metal::clamp(_e173, _e174, _e175);
    half _e179 = val;
    half _e181 = val;
    half _e184 = val;
    val = _e184 + metal::dot(metal::half2(_e179), metal::half2(_e181));
    half _e186 = val;
    half _e187 = val;
    half _e189 = val;
    val = _e189 + metal::max(_e186, _e187);
    half _e191 = val;
    half _e192 = val;
    half _e194 = val;
    val = _e194 + metal::min(_e191, _e192);
    half _e196 = val;
    half _e198 = val;
    val = _e198 + metal::sign(_e196);
    half _e201 = val;
    val = _e201 + 1.0h;
    metal::half2 _e205 = input_uniform.val_f16_2_;
    metal::float2 float_vec2_ = static_cast<metal::float2>(_e205);
    output.val_f16_2_ = static_cast<metal::half2>(float_vec2_);
    metal::half3 _e212 = input_uniform.val_f16_3_;
    metal::float3 float_vec3_ = static_cast<metal::float3>(_e212);
    output.val_f16_3_ = static_cast<metal::half3>(float_vec3_);
    metal::half4 _e219 = input_uniform.val_f16_4_;
    metal::float4 float_vec4_ = static_cast<metal::float4>(_e219);
    output.val_f16_4_ = static_cast<metal::half4>(float_vec4_);
    metal::half2x2 _e228 = input_uniform.val_mat2x2_;
    output.val_mat2x2_ = metal::half2x2(metal::float2x2(_e228));
    metal::half2x3 _e235 = input_uniform.val_mat2x3_;
    output.val_mat2x3_ = metal::half2x3(metal::float2x3(_e235));
    metal::half2x4 _e242 = input_uniform.val_mat2x4_;
    output.val_mat2x4_ = metal::half2x4(metal::float2x4(_e242));
    metal::half3x2 _e249 = input_uniform.val_mat3x2_;
    output.val_mat3x2_ = metal::half3x2(metal::float3x2(_e249));
    metal::half3x3 _e256 = input_uniform.val_mat3x3_;
    output.val_mat3x3_ = metal::half3x3(metal::float3x3(_e256));
    metal::half3x4 _e263 = input_uniform.val_mat3x4_;
    output.val_mat3x4_ = metal::half3x4(metal::float3x4(_e263));
    metal::half4x2 _e270 = input_uniform.val_mat4x2_;
    output.val_mat4x2_ = metal::half4x2(metal::float4x2(_e270));
    metal::half4x3 _e277 = input_uniform.val_mat4x3_;
    output.val_mat4x3_ = metal::half4x3(metal::float4x3(_e277));
    metal::half4x4 _e284 = input_uniform.val_mat4x4_;
    output.val_mat4x4_ = metal::half4x4(metal::float4x4(_e284));
    half _e287 = val;
    return _e287;
}

kernel void main_(
  constant UniformCompatible& input_uniform [[user(fake0)]]
, device UniformCompatible const& input_storage [[user(fake0)]]
, device StorageCompatible const& input_arrays [[user(fake0)]]
, device UniformCompatible& output [[user(fake0)]]
, device StorageCompatible& output_arrays [[user(fake0)]]
) {
    half private_variable = 1.0h;
    half _e3 = f16_function(2.0h, private_variable, input_uniform, input_storage, input_arrays, output, output_arrays);
    output.final_value = _e3;
    return;
}

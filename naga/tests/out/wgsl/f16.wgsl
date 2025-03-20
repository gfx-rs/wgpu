enable f16;

struct UniformCompatible {
    val_u32_: u32,
    val_i32_: i32,
    val_f32_: f32,
    val_f16_: f16,
    val_f16_2_: vec2<f16>,
    val_f16_3_: vec3<f16>,
    val_f16_4_: vec4<f16>,
    final_value: f16,
    val_mat2x2_: mat2x2<f16>,
    val_mat2x3_: mat2x3<f16>,
    val_mat2x4_: mat2x4<f16>,
    val_mat3x2_: mat3x2<f16>,
    val_mat3x3_: mat3x3<f16>,
    val_mat3x4_: mat3x4<f16>,
    val_mat4x2_: mat4x2<f16>,
    val_mat4x3_: mat4x3<f16>,
    val_mat4x4_: mat4x4<f16>,
}

struct StorageCompatible {
    val_f16_array_2_: array<f16, 2>,
}

struct LayoutTest {
    scalar1_: f16,
    scalar2_: f16,
    v3_: vec3<f16>,
    tuck_in: f16,
    scalar4_: f16,
    larger: u32,
}

const constant_variable: f16 = 15.203125h;

var<private> private_variable: f16 = 1h;
@group(0) @binding(0) 
var<uniform> input_uniform: UniformCompatible;
@group(0) @binding(1) 
var<storage> input_storage: UniformCompatible;
@group(0) @binding(2) 
var<storage> input_arrays: StorageCompatible;
@group(0) @binding(3) 
var<storage, read_write> output: UniformCompatible;
@group(0) @binding(4) 
var<storage, read_write> output_arrays: StorageCompatible;

fn f16_function(x: f16) -> f16 {
    var val: f16 = 15.203125h;

    let _e4 = val;
    val = (_e4 + -33344h);
    let _e6 = val;
    let _e9 = val;
    val = (_e9 + (_e6 + 5h));
    let _e13 = input_uniform.val_f32_;
    let _e14 = val;
    let _e18 = val;
    val = (_e18 + f16((_e13 + f32(_e14))));
    let _e22 = input_uniform.val_f16_;
    let _e25 = val;
    val = (_e25 + vec3(_e22).z);
    let _e31 = input_uniform.val_f16_;
    let _e34 = input_storage.val_f16_;
    output.val_f16_ = (_e31 + _e34);
    let _e40 = input_uniform.val_f16_2_;
    let _e43 = input_storage.val_f16_2_;
    output.val_f16_2_ = (_e40 + _e43);
    let _e49 = input_uniform.val_f16_3_;
    let _e52 = input_storage.val_f16_3_;
    output.val_f16_3_ = (_e49 + _e52);
    let _e58 = input_uniform.val_f16_4_;
    let _e61 = input_storage.val_f16_4_;
    output.val_f16_4_ = (_e58 + _e61);
    let _e67 = input_uniform.val_mat2x2_;
    let _e70 = input_storage.val_mat2x2_;
    output.val_mat2x2_ = (_e67 + _e70);
    let _e76 = input_uniform.val_mat2x3_;
    let _e79 = input_storage.val_mat2x3_;
    output.val_mat2x3_ = (_e76 + _e79);
    let _e85 = input_uniform.val_mat2x4_;
    let _e88 = input_storage.val_mat2x4_;
    output.val_mat2x4_ = (_e85 + _e88);
    let _e94 = input_uniform.val_mat3x2_;
    let _e97 = input_storage.val_mat3x2_;
    output.val_mat3x2_ = (_e94 + _e97);
    let _e103 = input_uniform.val_mat3x3_;
    let _e106 = input_storage.val_mat3x3_;
    output.val_mat3x3_ = (_e103 + _e106);
    let _e112 = input_uniform.val_mat3x4_;
    let _e115 = input_storage.val_mat3x4_;
    output.val_mat3x4_ = (_e112 + _e115);
    let _e121 = input_uniform.val_mat4x2_;
    let _e124 = input_storage.val_mat4x2_;
    output.val_mat4x2_ = (_e121 + _e124);
    let _e130 = input_uniform.val_mat4x3_;
    let _e133 = input_storage.val_mat4x3_;
    output.val_mat4x3_ = (_e130 + _e133);
    let _e139 = input_uniform.val_mat4x4_;
    let _e142 = input_storage.val_mat4x4_;
    output.val_mat4x4_ = (_e139 + _e142);
    let _e148 = input_arrays.val_f16_array_2_;
    output_arrays.val_f16_array_2_ = _e148;
    let _e149 = val;
    let _e151 = val;
    val = (_e151 + abs(_e149));
    let _e153 = val;
    let _e154 = val;
    let _e155 = val;
    let _e157 = val;
    val = (_e157 + clamp(_e153, _e154, _e155));
    let _e159 = val;
    let _e161 = val;
    let _e164 = val;
    val = (_e164 + dot(vec2(_e159), vec2(_e161)));
    let _e166 = val;
    let _e167 = val;
    let _e169 = val;
    val = (_e169 + max(_e166, _e167));
    let _e171 = val;
    let _e172 = val;
    let _e174 = val;
    val = (_e174 + min(_e171, _e172));
    let _e176 = val;
    let _e178 = val;
    val = (_e178 + sign(_e176));
    let _e181 = val;
    val = (_e181 + 1h);
    let _e185 = input_uniform.val_f16_2_;
    let float_vec2_ = vec2<f32>(_e185);
    output.val_f16_2_ = vec2<f16>(float_vec2_);
    let _e192 = input_uniform.val_f16_3_;
    let float_vec3_ = vec3<f32>(_e192);
    output.val_f16_3_ = vec3<f16>(float_vec3_);
    let _e199 = input_uniform.val_f16_4_;
    let float_vec4_ = vec4<f32>(_e199);
    output.val_f16_4_ = vec4<f16>(float_vec4_);
    let _e208 = input_uniform.val_mat2x2_;
    output.val_mat2x2_ = mat2x2<f16>(mat2x2<f32>(_e208));
    let _e215 = input_uniform.val_mat2x3_;
    output.val_mat2x3_ = mat2x3<f16>(mat2x3<f32>(_e215));
    let _e222 = input_uniform.val_mat2x4_;
    output.val_mat2x4_ = mat2x4<f16>(mat2x4<f32>(_e222));
    let _e229 = input_uniform.val_mat3x2_;
    output.val_mat3x2_ = mat3x2<f16>(mat3x2<f32>(_e229));
    let _e236 = input_uniform.val_mat3x3_;
    output.val_mat3x3_ = mat3x3<f16>(mat3x3<f32>(_e236));
    let _e243 = input_uniform.val_mat3x4_;
    output.val_mat3x4_ = mat3x4<f16>(mat3x4<f32>(_e243));
    let _e250 = input_uniform.val_mat4x2_;
    output.val_mat4x2_ = mat4x2<f16>(mat4x2<f32>(_e250));
    let _e257 = input_uniform.val_mat4x3_;
    output.val_mat4x3_ = mat4x3<f16>(mat4x3<f32>(_e257));
    let _e264 = input_uniform.val_mat4x4_;
    output.val_mat4x4_ = mat4x4<f16>(mat4x4<f32>(_e264));
    let _e267 = val;
    return _e267;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e3 = f16_function(2h);
    output.final_value = _e3;
    return;
}

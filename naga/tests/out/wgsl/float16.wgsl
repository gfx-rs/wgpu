struct UniformCompatible {
    val_u32_: u32,
    val_i32_: i32,
    val_f32_: f32,
    val_f16_: f16,
    val_f16_2_: vec2<f16>,
    val_f16_3_: vec3<f16>,
    val_f16_4_: vec4<f16>,
    final_value: f16,
}

struct StorageCompatible {
    val_f16_array_2_: array<f16, 2>,
    val_f16_array_2_1: array<f16, 2>,
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

    let _e6 = val;
    val = (_e6 + (1h - 33344h));
    let _e8 = val;
    let _e11 = val;
    val = (_e11 + (_e8 + 5h));
    let _e15 = input_uniform.val_f32_;
    let _e16 = val;
    let _e20 = val;
    val = (_e20 + f16((_e15 + f32(_e16))));
    let _e24 = input_uniform.val_f16_;
    let _e27 = val;
    val = (_e27 + vec3(_e24).z);
    let _e33 = input_uniform.val_f16_;
    let _e36 = input_storage.val_f16_;
    output.val_f16_ = (_e33 + _e36);
    let _e42 = input_uniform.val_f16_2_;
    let _e45 = input_storage.val_f16_2_;
    output.val_f16_2_ = (_e42 + _e45);
    let _e51 = input_uniform.val_f16_3_;
    let _e54 = input_storage.val_f16_3_;
    output.val_f16_3_ = (_e51 + _e54);
    let _e60 = input_uniform.val_f16_4_;
    let _e63 = input_storage.val_f16_4_;
    output.val_f16_4_ = (_e60 + _e63);
    let _e69 = input_arrays.val_f16_array_2_;
    output_arrays.val_f16_array_2_ = _e69;
    let _e70 = val;
    let _e72 = val;
    val = (_e72 + abs(_e70));
    let _e74 = val;
    let _e75 = val;
    let _e76 = val;
    let _e78 = val;
    val = (_e78 + clamp(_e74, _e75, _e76));
    let _e80 = val;
    let _e82 = val;
    let _e85 = val;
    val = (_e85 + dot(vec2(_e80), vec2(_e82)));
    let _e87 = val;
    let _e88 = val;
    let _e90 = val;
    val = (_e90 + max(_e87, _e88));
    let _e92 = val;
    let _e93 = val;
    let _e95 = val;
    val = (_e95 + min(_e92, _e93));
    let _e97 = val;
    let _e99 = val;
    val = (_e99 + sign(_e97));
    return 1h;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e3 = f16_function(2h);
    output.final_value = _e3;
    return;
}

struct UniformCompatible {
    val_u32_: u32,
    val_i32_: i32,
    val_f32_: f32,
    val_u64_: u64,
    val_u64_2_: vec2<u64>,
    val_u64_3_: vec3<u64>,
    val_u64_4_: vec4<u64>,
    val_i64_: i64,
    val_i64_2_: vec2<i64>,
    val_i64_3_: vec3<i64>,
    val_i64_4_: vec4<i64>,
    final_value: u64,
}

struct StorageCompatible {
    val_u64_array_2_: array<u64, 2>,
    val_i64_array_2_: array<i64, 2>,
}

const constant_variable: u64 = 20lu;

var<private> private_variable: i64 = 1li;
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

fn int64_function(x: i64) -> i64 {
    var val: i64 = 20li;

    let phony = private_variable;
    let _e5 = val;
    val = (_e5 + ((31li - 1002003004005006li) + -9223372036854775807li));
    let _e12 = val;
    let _e13 = val;
    val = (_e12 + (_e13 + 5li));
    let _e17 = val;
    let _e20 = input_uniform.val_u32_;
    let _e21 = val;
    val = (_e17 + i64((_e20 + u32(_e21))));
    let _e26 = val;
    let _e29 = input_uniform.val_i32_;
    let _e30 = val;
    val = (_e26 + i64((_e29 + i32(_e30))));
    let _e35 = val;
    let _e38 = input_uniform.val_f32_;
    let _e39 = val;
    val = (_e35 + i64((_e38 + f32(_e39))));
    let _e44 = val;
    let _e47 = input_uniform.val_i64_;
    val = (_e44 + vec3(_e47).z);
    let _e51 = val;
    let _e54 = input_uniform.val_u64_;
    val = (_e51 + bitcast<i64>(_e54));
    let _e57 = val;
    let _e60 = input_uniform.val_u64_2_;
    val = (_e57 + bitcast<vec2<i64>>(_e60).y);
    let _e64 = val;
    let _e67 = input_uniform.val_u64_3_;
    val = (_e64 + bitcast<vec3<i64>>(_e67).z);
    let _e71 = val;
    let _e74 = input_uniform.val_u64_4_;
    val = (_e71 + bitcast<vec4<i64>>(_e74).w);
    let _e78 = val;
    val = (_e78 + i64(-9223372036854775807 - 1));
    let _e85 = input_uniform.val_i64_;
    let _e88 = input_storage.val_i64_;
    output.val_i64_ = (_e85 + _e88);
    let _e94 = input_uniform.val_i64_2_;
    let _e97 = input_storage.val_i64_2_;
    output.val_i64_2_ = (_e94 + _e97);
    let _e103 = input_uniform.val_i64_3_;
    let _e106 = input_storage.val_i64_3_;
    output.val_i64_3_ = (_e103 + _e106);
    let _e112 = input_uniform.val_i64_4_;
    let _e115 = input_storage.val_i64_4_;
    output.val_i64_4_ = (_e112 + _e115);
    let _e121 = input_arrays.val_i64_array_2_;
    output_arrays.val_i64_array_2_ = _e121;
    let _e122 = val;
    let _e123 = val;
    val = (_e122 + abs(_e123));
    let _e126 = val;
    let _e127 = val;
    let _e128 = val;
    let _e129 = val;
    val = (_e126 + clamp(_e127, _e128, _e129));
    let _e132 = val;
    let _e133 = val;
    let _e135 = val;
    val = (_e132 + dot(vec2(_e133), vec2(_e135)));
    let _e139 = val;
    let _e140 = val;
    let _e141 = val;
    val = (_e139 + max(_e140, _e141));
    let _e144 = val;
    let _e145 = val;
    let _e146 = val;
    val = (_e144 + min(_e145, _e146));
    let _e149 = val;
    let _e150 = val;
    val = (_e149 + sign(_e150));
    let _e153 = val;
    return _e153;
}

fn uint64_function(x_1: u64) -> u64 {
    var val_1: u64 = 20lu;

    let _e3 = val_1;
    val_1 = (_e3 + ((31lu + 18446744073709551615lu) - 18446744073709551615lu));
    let _e10 = val_1;
    let _e11 = val_1;
    val_1 = (_e10 + (_e11 + 5lu));
    let _e15 = val_1;
    let _e18 = input_uniform.val_u32_;
    let _e19 = val_1;
    val_1 = (_e15 + u64((_e18 + u32(_e19))));
    let _e24 = val_1;
    let _e27 = input_uniform.val_i32_;
    let _e28 = val_1;
    val_1 = (_e24 + u64((_e27 + i32(_e28))));
    let _e33 = val_1;
    let _e36 = input_uniform.val_f32_;
    let _e37 = val_1;
    val_1 = (_e33 + u64((_e36 + f32(_e37))));
    let _e42 = val_1;
    let _e45 = input_uniform.val_u64_;
    val_1 = (_e42 + vec3(_e45).z);
    let _e49 = val_1;
    let _e52 = input_uniform.val_i64_;
    val_1 = (_e49 + bitcast<u64>(_e52));
    let _e55 = val_1;
    let _e58 = input_uniform.val_i64_2_;
    val_1 = (_e55 + bitcast<vec2<u64>>(_e58).y);
    let _e62 = val_1;
    let _e65 = input_uniform.val_i64_3_;
    val_1 = (_e62 + bitcast<vec3<u64>>(_e65).z);
    let _e69 = val_1;
    let _e72 = input_uniform.val_i64_4_;
    val_1 = (_e69 + bitcast<vec4<u64>>(_e72).w);
    let _e80 = input_uniform.val_u64_;
    let _e83 = input_storage.val_u64_;
    output.val_u64_ = (_e80 + _e83);
    let _e89 = input_uniform.val_u64_2_;
    let _e92 = input_storage.val_u64_2_;
    output.val_u64_2_ = (_e89 + _e92);
    let _e98 = input_uniform.val_u64_3_;
    let _e101 = input_storage.val_u64_3_;
    output.val_u64_3_ = (_e98 + _e101);
    let _e107 = input_uniform.val_u64_4_;
    let _e110 = input_storage.val_u64_4_;
    output.val_u64_4_ = (_e107 + _e110);
    let _e116 = input_arrays.val_u64_array_2_;
    output_arrays.val_u64_array_2_ = _e116;
    let _e117 = val_1;
    let _e118 = val_1;
    val_1 = (_e117 + abs(_e118));
    let _e121 = val_1;
    let _e122 = val_1;
    let _e123 = val_1;
    let _e124 = val_1;
    val_1 = (_e121 + clamp(_e122, _e123, _e124));
    let _e127 = val_1;
    let _e128 = val_1;
    let _e130 = val_1;
    val_1 = (_e127 + dot(vec2(_e128), vec2(_e130)));
    let _e134 = val_1;
    let _e135 = val_1;
    let _e136 = val_1;
    val_1 = (_e134 + max(_e135, _e136));
    let _e139 = val_1;
    let _e140 = val_1;
    let _e141 = val_1;
    val_1 = (_e139 + min(_e140, _e141));
    let _e144 = val_1;
    return _e144;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e3 = uint64_function(67lu);
    let _e5 = int64_function(60li);
    output.final_value = (_e3 + bitcast<u64>(_e5));
    return;
}

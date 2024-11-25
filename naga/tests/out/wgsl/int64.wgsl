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

    let _e8 = val;
    val = (_e8 + ((31li - 1002003004005006li) + -9223372036854775807li));
    let _e10 = val;
    let _e13 = val;
    val = (_e13 + (_e10 + 5li));
    let _e17 = input_uniform.val_u32_;
    let _e18 = val;
    let _e22 = val;
    val = (_e22 + i64((_e17 + u32(_e18))));
    let _e26 = input_uniform.val_i32_;
    let _e27 = val;
    let _e31 = val;
    val = (_e31 + i64((_e26 + i32(_e27))));
    let _e35 = input_uniform.val_f32_;
    let _e36 = val;
    let _e40 = val;
    val = (_e40 + i64((_e35 + f32(_e36))));
    let _e44 = input_uniform.val_i64_;
    let _e47 = val;
    val = (_e47 + vec3(_e44).z);
    let _e51 = input_uniform.val_u64_;
    let _e53 = val;
    val = (_e53 + bitcast<i64>(_e51));
    let _e57 = input_uniform.val_u64_2_;
    let _e60 = val;
    val = (_e60 + bitcast<vec2<i64>>(_e57).y);
    let _e64 = input_uniform.val_u64_3_;
    let _e67 = val;
    val = (_e67 + bitcast<vec3<i64>>(_e64).z);
    let _e71 = input_uniform.val_u64_4_;
    let _e74 = val;
    val = (_e74 + bitcast<vec4<i64>>(_e71).w);
    let _e80 = input_uniform.val_i64_;
    let _e83 = input_storage.val_i64_;
    output.val_i64_ = (_e80 + _e83);
    let _e89 = input_uniform.val_i64_2_;
    let _e92 = input_storage.val_i64_2_;
    output.val_i64_2_ = (_e89 + _e92);
    let _e98 = input_uniform.val_i64_3_;
    let _e101 = input_storage.val_i64_3_;
    output.val_i64_3_ = (_e98 + _e101);
    let _e107 = input_uniform.val_i64_4_;
    let _e110 = input_storage.val_i64_4_;
    output.val_i64_4_ = (_e107 + _e110);
    let _e116 = input_arrays.val_i64_array_2_;
    output_arrays.val_i64_array_2_ = _e116;
    let _e117 = val;
    let _e119 = val;
    val = (_e119 + abs(_e117));
    let _e121 = val;
    let _e122 = val;
    let _e123 = val;
    let _e125 = val;
    val = (_e125 + clamp(_e121, _e122, _e123));
    let _e127 = val;
    let _e129 = val;
    let _e132 = val;
    val = (_e132 + dot(vec2(_e127), vec2(_e129)));
    let _e134 = val;
    let _e135 = val;
    let _e137 = val;
    val = (_e137 + max(_e134, _e135));
    let _e139 = val;
    let _e140 = val;
    let _e142 = val;
    val = (_e142 + min(_e139, _e140));
    let _e144 = val;
    let _e146 = val;
    val = (_e146 + sign(_e144));
    let _e148 = val;
    return _e148;
}

fn uint64_function(x_1: u64) -> u64 {
    var val_1: u64 = 20lu;

    let _e8 = val_1;
    val_1 = (_e8 + ((31lu + 18446744073709551615lu) - 18446744073709551615lu));
    let _e10 = val_1;
    let _e13 = val_1;
    val_1 = (_e13 + (_e10 + 5lu));
    let _e17 = input_uniform.val_u32_;
    let _e18 = val_1;
    let _e22 = val_1;
    val_1 = (_e22 + u64((_e17 + u32(_e18))));
    let _e26 = input_uniform.val_i32_;
    let _e27 = val_1;
    let _e31 = val_1;
    val_1 = (_e31 + u64((_e26 + i32(_e27))));
    let _e35 = input_uniform.val_f32_;
    let _e36 = val_1;
    let _e40 = val_1;
    val_1 = (_e40 + u64((_e35 + f32(_e36))));
    let _e44 = input_uniform.val_u64_;
    let _e47 = val_1;
    val_1 = (_e47 + vec3(_e44).z);
    let _e51 = input_uniform.val_i64_;
    let _e53 = val_1;
    val_1 = (_e53 + bitcast<u64>(_e51));
    let _e57 = input_uniform.val_i64_2_;
    let _e60 = val_1;
    val_1 = (_e60 + bitcast<vec2<u64>>(_e57).y);
    let _e64 = input_uniform.val_i64_3_;
    let _e67 = val_1;
    val_1 = (_e67 + bitcast<vec3<u64>>(_e64).z);
    let _e71 = input_uniform.val_i64_4_;
    let _e74 = val_1;
    val_1 = (_e74 + bitcast<vec4<u64>>(_e71).w);
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
    let _e119 = val_1;
    val_1 = (_e119 + abs(_e117));
    let _e121 = val_1;
    let _e122 = val_1;
    let _e123 = val_1;
    let _e125 = val_1;
    val_1 = (_e125 + clamp(_e121, _e122, _e123));
    let _e127 = val_1;
    let _e129 = val_1;
    let _e132 = val_1;
    val_1 = (_e132 + dot(vec2(_e127), vec2(_e129)));
    let _e134 = val_1;
    let _e135 = val_1;
    let _e137 = val_1;
    val_1 = (_e137 + max(_e134, _e135));
    let _e139 = val_1;
    let _e140 = val_1;
    let _e142 = val_1;
    val_1 = (_e142 + min(_e139, _e140));
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

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
    let _e10 = val;
    val = (_e10 + ((31li - 1002003004005006li) + -9223372036854775807li));
    let _e12 = val;
    let _e15 = val;
    val = (_e15 + (_e12 + 5li));
    let _e19 = input_uniform.val_u32_;
    let _e20 = val;
    let _e24 = val;
    val = (_e24 + i64((_e19 + u32(_e20))));
    let _e28 = input_uniform.val_i32_;
    let _e29 = val;
    let _e33 = val;
    val = (_e33 + i64((_e28 + i32(_e29))));
    let _e37 = input_uniform.val_f32_;
    let _e38 = val;
    let _e42 = val;
    val = (_e42 + i64((_e37 + f32(_e38))));
    let _e46 = input_uniform.val_i64_;
    let _e49 = val;
    val = (_e49 + vec3(_e46).z);
    let _e53 = input_uniform.val_u64_;
    let _e55 = val;
    val = (_e55 + bitcast<i64>(_e53));
    let _e59 = input_uniform.val_u64_2_;
    let _e62 = val;
    val = (_e62 + bitcast<vec2<i64>>(_e59).y);
    let _e66 = input_uniform.val_u64_3_;
    let _e69 = val;
    val = (_e69 + bitcast<vec3<i64>>(_e66).z);
    let _e73 = input_uniform.val_u64_4_;
    let _e76 = val;
    val = (_e76 + bitcast<vec4<i64>>(_e73).w);
    let _e79 = val;
    val = (_e79 + i64(-9223372036854775807 - 1));
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
    let _e124 = val;
    val = (_e124 + abs(_e122));
    let _e126 = val;
    let _e127 = val;
    let _e128 = val;
    let _e130 = val;
    val = (_e130 + clamp(_e126, _e127, _e128));
    let _e132 = val;
    let _e134 = val;
    let _e137 = val;
    val = (_e137 + dot(vec2(_e132), vec2(_e134)));
    let _e139 = val;
    let _e140 = val;
    let _e142 = val;
    val = (_e142 + max(_e139, _e140));
    let _e144 = val;
    let _e145 = val;
    let _e147 = val;
    val = (_e147 + min(_e144, _e145));
    let _e149 = val;
    let _e151 = val;
    val = (_e151 + sign(_e149));
    let _e153 = val;
    return _e153;
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

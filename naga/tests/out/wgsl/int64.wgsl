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

    let _e6 = val;
    val = (_e6 + (31li - 1002003004005006li));
    let _e8 = val;
    let _e11 = val;
    val = (_e11 + (_e8 + 5li));
    let _e15 = input_uniform.val_u32_;
    let _e16 = val;
    let _e20 = val;
    val = (_e20 + i64((_e15 + u32(_e16))));
    let _e24 = input_uniform.val_i32_;
    let _e25 = val;
    let _e29 = val;
    val = (_e29 + i64((_e24 + i32(_e25))));
    let _e33 = input_uniform.val_f32_;
    let _e34 = val;
    let _e38 = val;
    val = (_e38 + i64((_e33 + f32(_e34))));
    let _e42 = input_uniform.val_i64_;
    let _e45 = val;
    val = (_e45 + vec3(_e42).z);
    let _e49 = input_uniform.val_u64_;
    let _e51 = val;
    val = (_e51 + bitcast<i64>(_e49));
    let _e55 = input_uniform.val_u64_2_;
    let _e58 = val;
    val = (_e58 + bitcast<vec2<i64>>(_e55).y);
    let _e62 = input_uniform.val_u64_3_;
    let _e65 = val;
    val = (_e65 + bitcast<vec3<i64>>(_e62).z);
    let _e69 = input_uniform.val_u64_4_;
    let _e72 = val;
    val = (_e72 + bitcast<vec4<i64>>(_e69).w);
    let _e78 = input_uniform.val_i64_;
    let _e81 = input_storage.val_i64_;
    output.val_i64_ = (_e78 + _e81);
    let _e87 = input_uniform.val_i64_2_;
    let _e90 = input_storage.val_i64_2_;
    output.val_i64_2_ = (_e87 + _e90);
    let _e96 = input_uniform.val_i64_3_;
    let _e99 = input_storage.val_i64_3_;
    output.val_i64_3_ = (_e96 + _e99);
    let _e105 = input_uniform.val_i64_4_;
    let _e108 = input_storage.val_i64_4_;
    output.val_i64_4_ = (_e105 + _e108);
    let _e114 = input_arrays.val_i64_array_2_;
    output_arrays.val_i64_array_2_ = _e114;
    let _e115 = val;
    let _e117 = val;
    val = (_e117 + abs(_e115));
    let _e119 = val;
    let _e120 = val;
    let _e121 = val;
    let _e123 = val;
    val = (_e123 + clamp(_e119, _e120, _e121));
    let _e125 = val;
    let _e127 = val;
    let _e130 = val;
    val = (_e130 + dot(vec2(_e125), vec2(_e127)));
    let _e132 = val;
    let _e133 = val;
    let _e135 = val;
    val = (_e135 + max(_e132, _e133));
    let _e137 = val;
    let _e138 = val;
    let _e140 = val;
    val = (_e140 + min(_e137, _e138));
    let _e142 = val;
    let _e144 = val;
    val = (_e144 + sign(_e142));
    let _e146 = val;
    return _e146;
}

fn uint64_function(x_1: u64) -> u64 {
    var val_1: u64 = 20lu;

    let _e6 = val_1;
    val_1 = (_e6 + (31lu + 1002003004005006lu));
    let _e8 = val_1;
    let _e11 = val_1;
    val_1 = (_e11 + (_e8 + 5lu));
    let _e15 = input_uniform.val_u32_;
    let _e16 = val_1;
    let _e20 = val_1;
    val_1 = (_e20 + u64((_e15 + u32(_e16))));
    let _e24 = input_uniform.val_i32_;
    let _e25 = val_1;
    let _e29 = val_1;
    val_1 = (_e29 + u64((_e24 + i32(_e25))));
    let _e33 = input_uniform.val_f32_;
    let _e34 = val_1;
    let _e38 = val_1;
    val_1 = (_e38 + u64((_e33 + f32(_e34))));
    let _e42 = input_uniform.val_u64_;
    let _e45 = val_1;
    val_1 = (_e45 + vec3(_e42).z);
    let _e49 = input_uniform.val_i64_;
    let _e51 = val_1;
    val_1 = (_e51 + bitcast<u64>(_e49));
    let _e55 = input_uniform.val_i64_2_;
    let _e58 = val_1;
    val_1 = (_e58 + bitcast<vec2<u64>>(_e55).y);
    let _e62 = input_uniform.val_i64_3_;
    let _e65 = val_1;
    val_1 = (_e65 + bitcast<vec3<u64>>(_e62).z);
    let _e69 = input_uniform.val_i64_4_;
    let _e72 = val_1;
    val_1 = (_e72 + bitcast<vec4<u64>>(_e69).w);
    let _e78 = input_uniform.val_u64_;
    let _e81 = input_storage.val_u64_;
    output.val_u64_ = (_e78 + _e81);
    let _e87 = input_uniform.val_u64_2_;
    let _e90 = input_storage.val_u64_2_;
    output.val_u64_2_ = (_e87 + _e90);
    let _e96 = input_uniform.val_u64_3_;
    let _e99 = input_storage.val_u64_3_;
    output.val_u64_3_ = (_e96 + _e99);
    let _e105 = input_uniform.val_u64_4_;
    let _e108 = input_storage.val_u64_4_;
    output.val_u64_4_ = (_e105 + _e108);
    let _e114 = input_arrays.val_u64_array_2_;
    output_arrays.val_u64_array_2_ = _e114;
    let _e115 = val_1;
    let _e117 = val_1;
    val_1 = (_e117 + abs(_e115));
    let _e119 = val_1;
    let _e120 = val_1;
    let _e121 = val_1;
    let _e123 = val_1;
    val_1 = (_e123 + clamp(_e119, _e120, _e121));
    let _e125 = val_1;
    let _e127 = val_1;
    let _e130 = val_1;
    val_1 = (_e130 + dot(vec2(_e125), vec2(_e127)));
    let _e132 = val_1;
    let _e133 = val_1;
    let _e135 = val_1;
    val_1 = (_e135 + max(_e132, _e133));
    let _e137 = val_1;
    let _e138 = val_1;
    let _e140 = val_1;
    val_1 = (_e140 + min(_e137, _e138));
    let _e142 = val_1;
    return _e142;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e3 = uint64_function(67lu);
    let _e5 = int64_function(60li);
    output.final_value = (_e3 + bitcast<u64>(_e5));
    return;
}

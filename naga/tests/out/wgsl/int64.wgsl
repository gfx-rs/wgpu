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
    let val_1_ = (31li - 1002003004005006li);
    let val_2_ = (val_1_ + 5li);
    let _e8 = input_uniform.val_u32_;
    let _e14 = input_uniform.val_i32_;
    let _e21 = input_uniform.val_f32_;
    let val_3_ = ((i64((_e8 + u32(val_2_))) + i64((_e14 + i32(val_2_)))) + i64((_e21 + f32(val_2_))));
    let _e28 = input_uniform.val_i64_;
    let val_4_ = vec3(_e28);
    let _e32 = input_uniform.val_u64_;
    let val_5_ = bitcast<i64>(_e32);
    let _e36 = input_uniform.val_u64_2_;
    let val_6_ = bitcast<vec2<i64>>(_e36);
    let _e40 = input_uniform.val_u64_3_;
    let val_7_ = bitcast<vec3<i64>>(_e40);
    let _e44 = input_uniform.val_u64_4_;
    let val_8_ = bitcast<vec4<i64>>(_e44);
    let _e50 = input_uniform.val_i64_;
    let _e53 = input_storage.val_i64_;
    output.val_i64_ = (_e50 + _e53);
    let _e59 = input_uniform.val_i64_2_;
    let _e62 = input_storage.val_i64_2_;
    output.val_i64_2_ = (_e59 + _e62);
    let _e68 = input_uniform.val_i64_3_;
    let _e71 = input_storage.val_i64_3_;
    output.val_i64_3_ = (_e68 + _e71);
    let _e77 = input_uniform.val_i64_4_;
    let _e80 = input_storage.val_i64_4_;
    output.val_i64_4_ = (_e77 + _e80);
    let _e86 = input_arrays.val_i64_array_2_;
    output_arrays.val_i64_array_2_ = _e86;
    return (((((((((val_1_ + val_2_) + val_3_) + val_4_.x) + val_5_) + val_6_.x) + val_7_.x) + val_8_.x) + 20li) + 50li);
}

fn uint64_function(x_1: u64) -> u64 {
    let val_1_1 = (31lu + 1002003004005006lu);
    let val_2_1 = (val_1_1 + 5lu);
    let _e8 = input_uniform.val_u32_;
    let _e14 = input_uniform.val_i32_;
    let _e21 = input_uniform.val_f32_;
    let val_3_1 = ((u64((_e8 + u32(val_2_1))) + u64((_e14 + i32(val_2_1)))) + u64((_e21 + f32(val_2_1))));
    let _e28 = input_uniform.val_u64_;
    let val_4_1 = vec3(_e28);
    let _e32 = input_uniform.val_i64_;
    let val_5_1 = bitcast<u64>(_e32);
    let _e36 = input_uniform.val_i64_2_;
    let val_6_1 = bitcast<vec2<u64>>(_e36);
    let _e40 = input_uniform.val_i64_3_;
    let val_7_1 = bitcast<vec3<u64>>(_e40);
    let _e44 = input_uniform.val_i64_4_;
    let val_8_1 = bitcast<vec4<u64>>(_e44);
    let _e50 = input_uniform.val_u64_;
    let _e53 = input_storage.val_u64_;
    output.val_u64_ = (_e50 + _e53);
    let _e59 = input_uniform.val_u64_2_;
    let _e62 = input_storage.val_u64_2_;
    output.val_u64_2_ = (_e59 + _e62);
    let _e68 = input_uniform.val_u64_3_;
    let _e71 = input_storage.val_u64_3_;
    output.val_u64_3_ = (_e68 + _e71);
    let _e77 = input_uniform.val_u64_4_;
    let _e80 = input_storage.val_u64_4_;
    output.val_u64_4_ = (_e77 + _e80);
    let _e86 = input_arrays.val_u64_array_2_;
    output_arrays.val_u64_array_2_ = _e86;
    return (((((((((val_1_1 + val_2_1) + val_3_1) + val_4_1.x) + val_5_1) + val_6_1.x) + val_7_1.x) + val_8_1.x) + 20lu) + 50lu);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e1 = uint64_function(67lu);
    let _e3 = int64_function(60li);
    return;
}

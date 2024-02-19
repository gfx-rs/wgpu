struct UniformCompatible {
    val_u32_: u32,
    val_i32_: i32,
    val_f32_: f32,
    val_f64_: f64,
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
    let _e12 = input_uniform.val_i32_;
    let _e17 = input_uniform.val_f32_;
    let _e22 = input_uniform.val_f64_;
    let val_3_ = (((i64(_e8) + i64(_e12)) + i64(_e17)) + i64(_e22));
    let _e27 = input_uniform.val_i64_;
    let val_4_ = vec3(_e27);
    let _e33 = input_uniform.val_i64_;
    let _e36 = input_storage.val_i64_;
    output.val_i64_ = (_e33 + _e36);
    let _e42 = input_uniform.val_i64_2_;
    let _e45 = input_storage.val_i64_2_;
    output.val_i64_2_ = (_e42 + _e45);
    let _e51 = input_uniform.val_i64_3_;
    let _e54 = input_storage.val_i64_3_;
    output.val_i64_3_ = (_e51 + _e54);
    let _e60 = input_uniform.val_i64_4_;
    let _e63 = input_storage.val_i64_4_;
    output.val_i64_4_ = (_e60 + _e63);
    let _e69 = input_arrays.val_i64_array_2_;
    output_arrays.val_i64_array_2_ = _e69;
    return ((((val_1_ + val_2_) + val_3_) + 20li) + 50li);
}

fn uint64_function(x_1: u64) -> u64 {
    let val_1_1 = (31lu + 1002003004005006lu);
    let val_2_1 = (val_1_1 + 5lu);
    let _e8 = input_uniform.val_u32_;
    let _e12 = input_uniform.val_i32_;
    let _e17 = input_uniform.val_f32_;
    let _e22 = input_uniform.val_f64_;
    let val_3_1 = (((u64(_e8) + u64(_e12)) + u64(_e17)) + u64(_e22));
    let _e27 = input_uniform.val_u64_;
    let val_4_1 = vec3(_e27);
    let _e33 = input_uniform.val_u64_;
    let _e36 = input_storage.val_u64_;
    output.val_u64_ = (_e33 + _e36);
    let _e42 = input_uniform.val_u64_2_;
    let _e45 = input_storage.val_u64_2_;
    output.val_u64_2_ = (_e42 + _e45);
    let _e51 = input_uniform.val_u64_3_;
    let _e54 = input_storage.val_u64_3_;
    output.val_u64_3_ = (_e51 + _e54);
    let _e60 = input_uniform.val_u64_4_;
    let _e63 = input_storage.val_u64_4_;
    output.val_u64_4_ = (_e60 + _e63);
    let _e69 = input_arrays.val_u64_array_2_;
    output_arrays.val_u64_array_2_ = _e69;
    return ((((val_1_1 + val_2_1) + val_3_1) + 20lu) + 50lu);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e1 = uint64_function(67lu);
    let _e3 = int64_function(60li);
    return;
}

enable f16;

const MIN_F16_: f16 = -65504h;
const MAX_F16_: f16 = 65504h;
const MIN_F32_: f32 = -340282350000000000000000000000000000000f;
const MAX_F32_: f32 = 340282350000000000000000000000000000000f;
const MIN_F64_: f64 = -1.7976931348623157e308lf;
const MAX_F64_: f64 = 1.7976931348623157e308lf;

fn test_const_eval() {
    var min_f16_to_i32_: i32 = -65504i;
    var max_f16_to_i32_: i32 = 65504i;
    var min_f16_to_u32_: u32 = 0u;
    var max_f16_to_u32_: u32 = 65504u;
    var min_f16_to_i64_: i64 = -65504li;
    var max_f16_to_i64_: i64 = 65504li;
    var min_f16_to_u64_: u64 = 0lu;
    var max_f16_to_u64_: u64 = 65504lu;
    var min_f32_to_i32_: i32 = i32(-2147483648);
    var max_f32_to_i32_: i32 = 2147483520i;
    var min_f32_to_u32_: u32 = 0u;
    var max_f32_to_u32_: u32 = 4294967040u;
    var min_f32_to_i64_: i64 = i64(-9223372036854775807 - 1);
    var max_f32_to_i64_: i64 = 9223371487098961920li;
    var min_f32_to_u64_: u64 = 0lu;
    var max_f32_to_u64_: u64 = 18446742974197923840lu;
    var min_f64_to_i64_: i64 = i64(-9223372036854775807 - 1);
    var max_f64_to_i64_: i64 = 9223372036854774784li;
    var min_f64_to_u64_: u64 = 0lu;
    var max_f64_to_u64_: u64 = 18446744073709549568lu;
    var min_abstract_float_to_i32_: i32 = i32(-2147483648);
    var max_abstract_float_to_i32_: i32 = 2147483647i;
    var min_abstract_float_to_u32_: u32 = 0u;
    var max_abstract_float_to_u32_: u32 = 4294967295u;
    var min_abstract_float_to_i64_: i64 = i64(-9223372036854775807 - 1);
    var max_abstract_float_to_i64_: i64 = 9223372036854774784li;
    var min_abstract_float_to_u64_: u64 = 0lu;
    var max_abstract_float_to_u64_: u64 = 18446744073709549568lu;

    return;
}

fn test_f16_to_i32_(f: f16) -> i32 {
    return i32(f);
}

fn test_f16_to_u32_(f_1: f16) -> u32 {
    return u32(f_1);
}

fn test_f16_to_i64_(f_2: f16) -> i64 {
    return i64(f_2);
}

fn test_f16_to_u64_(f_3: f16) -> u64 {
    return u64(f_3);
}

fn test_f32_to_i32_(f_4: f32) -> i32 {
    return i32(f_4);
}

fn test_f32_to_u32_(f_5: f32) -> u32 {
    return u32(f_5);
}

fn test_f32_to_i64_(f_6: f32) -> i64 {
    return i64(f_6);
}

fn test_f32_to_u64_(f_7: f32) -> u64 {
    return u64(f_7);
}

fn test_f64_to_i32_(f_8: f64) -> i32 {
    return i32(f_8);
}

fn test_f64_to_u32_(f_9: f64) -> u32 {
    return u32(f_9);
}

fn test_f64_to_i64_(f_10: f64) -> i64 {
    return i64(f_10);
}

fn test_f64_to_u64_(f_11: f64) -> u64 {
    return u64(f_11);
}

fn test_f16_to_i32_vec(f_12: vec2<f16>) -> vec2<i32> {
    return vec2<i32>(f_12);
}

fn test_f16_to_u32_vec(f_13: vec2<f16>) -> vec2<u32> {
    return vec2<u32>(f_13);
}

fn test_f16_to_i64_vec(f_14: vec2<f16>) -> vec2<i64> {
    return vec2<i64>(f_14);
}

fn test_f16_to_u64_vec(f_15: vec2<f16>) -> vec2<u64> {
    return vec2<u64>(f_15);
}

fn test_f32_to_i32_vec(f_16: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(f_16);
}

fn test_f32_to_u32_vec(f_17: vec2<f32>) -> vec2<u32> {
    return vec2<u32>(f_17);
}

fn test_f32_to_i64_vec(f_18: vec2<f32>) -> vec2<i64> {
    return vec2<i64>(f_18);
}

fn test_f32_to_u64_vec(f_19: vec2<f32>) -> vec2<u64> {
    return vec2<u64>(f_19);
}

fn test_f64_to_i32_vec(f_20: vec2<f64>) -> vec2<i32> {
    return vec2<i32>(f_20);
}

fn test_f64_to_u32_vec(f_21: vec2<f64>) -> vec2<u32> {
    return vec2<u32>(f_21);
}

fn test_f64_to_i64_vec(f_22: vec2<f64>) -> vec2<i64> {
    return vec2<i64>(f_22);
}

fn test_f64_to_u64_vec(f_23: vec2<f64>) -> vec2<u64> {
    return vec2<u64>(f_23);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    test_const_eval();
    let _e1 = test_f16_to_i32_(1h);
    let _e3 = test_f16_to_u32_(1h);
    let _e5 = test_f16_to_i64_(1h);
    let _e7 = test_f16_to_u64_(1h);
    let _e9 = test_f32_to_i32_(1f);
    let _e11 = test_f32_to_u32_(1f);
    let _e13 = test_f32_to_i64_(1f);
    let _e15 = test_f32_to_u64_(1f);
    let _e17 = test_f64_to_i32_(1.0lf);
    let _e19 = test_f64_to_u32_(1.0lf);
    let _e21 = test_f64_to_i64_(1.0lf);
    let _e23 = test_f64_to_u64_(1.0lf);
    let _e27 = test_f16_to_i32_vec(vec2<f16>(1h, 2h));
    let _e31 = test_f16_to_u32_vec(vec2<f16>(1h, 2h));
    let _e35 = test_f16_to_i64_vec(vec2<f16>(1h, 2h));
    let _e39 = test_f16_to_u64_vec(vec2<f16>(1h, 2h));
    let _e43 = test_f32_to_i32_vec(vec2<f32>(1f, 2f));
    let _e47 = test_f32_to_u32_vec(vec2<f32>(1f, 2f));
    let _e51 = test_f32_to_i64_vec(vec2<f32>(1f, 2f));
    let _e55 = test_f32_to_u64_vec(vec2<f32>(1f, 2f));
    let _e59 = test_f64_to_i32_vec(vec2<f64>(1.0lf, 2.0lf));
    let _e63 = test_f64_to_u32_vec(vec2<f64>(1.0lf, 2.0lf));
    let _e67 = test_f64_to_i64_vec(vec2<f64>(1.0lf, 2.0lf));
    let _e71 = test_f64_to_u64_vec(vec2<f64>(1.0lf, 2.0lf));
    return;
}

enable f16;

const MIN_F16 = -65504h;
const MAX_F16 = 65504h;
const MIN_F32 = -3.40282347E+38f;
const MAX_F32 = 3.40282347E+38f;
const MIN_F64 = -1.7976931348623157E+308lf;
const MAX_F64 = 1.7976931348623157E+308lf;
const MIN_ABSTRACT_FLOAT = -1.7976931348623157E+308;
const MAX_ABSTRACT_FLOAT = 1.7976931348623157E+308;

// conversion from float to int during const evaluation. If value is out of
// range of the destination type we must clamp to a value that is representable
// by both the source and destination types.
fn test_const_eval() {
    var min_f16_to_i32 = i32(MIN_F16);
    var max_f16_to_i32 = i32(MAX_F16);
    var min_f16_to_u32 = u32(MIN_F16);
    var max_f16_to_u32 = u32(MAX_F16);
    var min_f16_to_i64 = i64(MIN_F16);
    var max_f16_to_i64 = i64(MAX_F16);
    var min_f16_to_u64 = u64(MIN_F16);
    var max_f16_to_u64 = u64(MAX_F16);
    var min_f32_to_i32 = i32(MIN_F32);
    var max_f32_to_i32 = i32(MAX_F32);
    var min_f32_to_u32 = u32(MIN_F32);
    var max_f32_to_u32 = u32(MAX_F32);
    var min_f32_to_i64 = i64(MIN_F32);
    var max_f32_to_i64 = i64(MAX_F32);
    var min_f32_to_u64 = u64(MIN_F32);
    var max_f32_to_u64 = u64(MAX_F32);
    var min_f64_to_i64 = i64(MIN_F64);
    var max_f64_to_i64 = i64(MAX_F64);
    var min_f64_to_u64 = u64(MIN_F64);
    var max_f64_to_u64 = u64(MAX_F64);
    var min_abstract_float_to_i32 = i32(MIN_ABSTRACT_FLOAT);
    var max_abstract_float_to_i32 = i32(MAX_ABSTRACT_FLOAT);
    var min_abstract_float_to_u32 = u32(MIN_ABSTRACT_FLOAT);
    var max_abstract_float_to_u32 = u32(MAX_ABSTRACT_FLOAT);
    var min_abstract_float_to_i64 = i64(MIN_ABSTRACT_FLOAT);
    var max_abstract_float_to_i64 = i64(MAX_ABSTRACT_FLOAT);
    var min_abstract_float_to_u64 = u64(MIN_ABSTRACT_FLOAT);
    var max_abstract_float_to_u64 = u64(MAX_ABSTRACT_FLOAT);
}

// conversion from float to int at runtime. Generated code must ensure we avoid
// undefined behaviour due to casting a value that is out of range of the
// destination type, and that in such cases the result is a value that is
// representable by both the source and destination types.
fn test_f16_to_i32(f: f16) -> i32 {
  return i32(f);
}

fn test_f16_to_u32(f: f16) -> u32 {
  return u32(f);
}

fn test_f16_to_i64(f: f16) -> i64 {
  return i64(f);
}

fn test_f16_to_u64(f: f16) -> u64 {
  return u64(f);
}

fn test_f32_to_i32(f: f32) -> i32 {
  return i32(f);
}

fn test_f32_to_u32(f: f32) -> u32 {
  return u32(f);
}

fn test_f32_to_i64(f: f32) -> i64 {
  return i64(f);
}

fn test_f32_to_u64(f: f32) -> u64 {
  return u64(f);
}

fn test_f64_to_i32(f: f64) -> i32 {
  return i32(f);
}

fn test_f64_to_u32(f: f64) -> u32 {
  return u32(f);
}

fn test_f64_to_i64(f: f64) -> i64 {
  return i64(f);
}

fn test_f64_to_u64(f: f64) -> u64 {
  return u64(f);
}

fn test_f16_to_i32_vec(f: vec2<f16>) -> vec2<i32> {
  return vec2<i32>(f);
}

fn test_f16_to_u32_vec(f: vec2<f16>) -> vec2<u32> {
  return vec2<u32>(f);
}

fn test_f16_to_i64_vec(f: vec2<f16>) -> vec2<i64> {
  return vec2<i64>(f);
}

fn test_f16_to_u64_vec(f: vec2<f16>) -> vec2<u64> {
  return vec2<u64>(f);
}

fn test_f32_to_i32_vec(f: vec2<f32>) -> vec2<i32> {
  return vec2<i32>(f);
}

fn test_f32_to_u32_vec(f: vec2<f32>) -> vec2<u32> {
  return vec2<u32>(f);
}

fn test_f32_to_i64_vec(f: vec2<f32>) -> vec2<i64> {
  return vec2<i64>(f);
}

fn test_f32_to_u64_vec(f: vec2<f32>) -> vec2<u64> {
  return vec2<u64>(f);
}

fn test_f64_to_i32_vec(f: vec2<f64>) -> vec2<i32> {
  return vec2<i32>(f);
}

fn test_f64_to_u32_vec(f: vec2<f64>) -> vec2<u32> {
  return vec2<u32>(f);
}

fn test_f64_to_i64_vec(f: vec2<f64>) -> vec2<i64> {
  return vec2<i64>(f);
}

fn test_f64_to_u64_vec(f: vec2<f64>) -> vec2<u64> {
  return vec2<u64>(f);
}

@compute @workgroup_size(1)
fn main() {
    test_const_eval();
    test_f16_to_i32(1.);
    test_f16_to_u32(1.);
    test_f16_to_i64(1.);
    test_f16_to_u64(1.);
    test_f32_to_i32(1.);
    test_f32_to_u32(1.);
    test_f32_to_i64(1.);
    test_f32_to_u64(1.);
    test_f64_to_i32(1.);
    test_f64_to_u32(1.);
    test_f64_to_i64(1.);
    test_f64_to_u64(1.);
    test_f16_to_i32_vec(vec2(1., 2.));
    test_f16_to_u32_vec(vec2(1., 2.));
    test_f16_to_i64_vec(vec2(1., 2.));
    test_f16_to_u64_vec(vec2(1., 2.));
    test_f32_to_i32_vec(vec2(1., 2.));
    test_f32_to_u32_vec(vec2(1., 2.));
    test_f32_to_i64_vec(vec2(1., 2.));
    test_f32_to_u64_vec(vec2(1., 2.));
    test_f64_to_i32_vec(vec2(1., 2.));
    test_f64_to_u32_vec(vec2(1., 2.));
    test_f64_to_i64_vec(vec2(1., 2.));
    test_f64_to_u64_vec(vec2(1., 2.));
}

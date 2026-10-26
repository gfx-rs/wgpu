// conversion-float-to-int-glsl.wgsl test with `f64` types removed, so it can be
// used with GLSL ES 3.00 (WebGL2).

fn test_f32_to_i32(f: f32) -> i32 {
  return i32(f);
}

fn test_f32_to_u32(f: f32) -> u32 {
  return u32(f);
}

fn test_f32_to_i32_vec(f: vec2<f32>) -> vec2<i32> {
  return vec2<i32>(f);
}

fn test_f32_to_u32_vec(f: vec2<f32>) -> vec2<u32> {
  return vec2<u32>(f);
}

// The clamp also keeps `u32(f32(x))` from being folded to `x`.
fn test_u32_to_f32_to_u32(x: u32) -> u32 {
  return u32(f32(x));
}

// Conversions to float and bitcasts are not affected.
fn test_unaffected(i: i32, u: u32, f: f32) -> vec4<f32> {
  return vec4<f32>(f32(i), f32(u), bitcast<f32>(i), f32(bitcast<i32>(f)));
}

@fragment
fn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let i = test_f32_to_i32(pos.x) + test_f32_to_i32_vec(pos.xy).y;
  let u = test_f32_to_u32(pos.y) + test_f32_to_u32_vec(pos.xy).x
    + test_u32_to_f32_to_u32(u32(pos.z));
  return test_unaffected(i, u, pos.w);
}

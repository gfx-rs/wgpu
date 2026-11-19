// https://github.com/gfx-rs/wgpu/issues/10093
// GLSL ES 3.00 has no integer `mix`, so this is lowered componentwise.
@fragment
fn main(
    @location(0) @interpolate(flat) a: vec4<i32>,
    @location(1) @interpolate(flat) b: vec4<i32>,
) -> @location(0) vec4<i32> {
    return select(a * 2, b + 7, a < b);
}

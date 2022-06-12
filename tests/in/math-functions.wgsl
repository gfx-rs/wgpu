@vertex
fn main() {
    let f = 1.0;
    let v = vec4<f32>(0.0);
    let a = degrees(f);
    let b = radians(f);
    let c = degrees(v);
    let d = radians(v);
    let const_dot = dot(vec2<i32>(), vec2<i32>());
    let first_leading_bit_abs = firstLeadingBit(abs(0u));
}

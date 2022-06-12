@vertex 
fn main() {
    let v = vec4<f32>(0.0);
    let a = degrees(1.0);
    let b = radians(1.0);
    let c = degrees(v);
    let d = radians(v);
    let const_dot = dot(vec2<i32>(0, 0), vec2<i32>(0, 0));
    let first_leading_bit_abs = firstLeadingBit(abs(0u));
}

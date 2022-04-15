// Standard functions.

@fragment
fn derivatives(@builtin(position) foo: vec4<f32>) -> @location(0) vec4<f32> {
    let x = dpdx(foo);
    let y = dpdy(foo);
    let z = fwidth(foo);
    return (x + y) * z;
}

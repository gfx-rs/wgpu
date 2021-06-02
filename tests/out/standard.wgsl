[[stage(fragment)]]
fn derivatives([[builtin(position)]] foo: vec4<f32>) -> [[location(0)]] vec4<f32> {
    let x: vec4<f32> = dpdx(foo);
    let y: vec4<f32> = dpdy(foo);
    let z: vec4<f32> = fwidth(foo);
    return ((x + y) * z);
}

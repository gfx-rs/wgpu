[[stage(fragment)]]
fn derivatives([[builtin(position)]] foo: vec4<f32>) -> [[location(0)]] vec4<f32> {
    let _e1: vec4<f32> = dpdx(foo);
    let _e2: vec4<f32> = dpdy(foo);
    let _e3: vec4<f32> = fwidth(foo);
    return ((_e1 + _e2) * _e3);
}

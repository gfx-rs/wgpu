fn main1() {
    var x: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);

    let e3: vec3<f32> = x;
    let e8: vec2<f32> = vec2<f32>(3.0, 4.0);
    x.z = e8.x;
    x.x = e8.y;
    let e13: vec3<f32> = x;
    let e15: vec3<f32> = x;
    let e18: vec2<f32> = (e15.xy * 5.0);
    x.x = e18.x;
    x.y = e18.y;
    return;
}

[[stage(fragment)]]
fn main() {
    main1();
    return;
}

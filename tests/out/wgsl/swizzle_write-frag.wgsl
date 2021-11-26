fn main_1() {
    var x: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);

    let _e3 = x;
    let _e8 = vec2<f32>(3.0, 4.0);
    x.z = _e8.x;
    x.x = _e8.y;
    let _e13 = x;
    let _e15 = x;
    let _e18 = (_e15.xy * 5.0);
    x.x = _e18.x;
    x.y = _e18.y;
    return;
}

[[stage(fragment)]]
fn main() {
    main_1();
    return;
}

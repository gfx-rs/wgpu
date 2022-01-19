fn foo(p: ptr<function, vec2<f32>>) {
    return;
}

fn main_1() {
    var x: vec3<f32> = vec3<f32>(2.0, 2.0, 2.0);
    var local: vec2<f32>;

    let _e3 = x;
    let _e8 = vec2<f32>(3.0, 4.0);
    x.z = _e8.x;
    x.x = _e8.y;
    let _e13 = x;
    let _e15 = x;
    let _e18 = (_e15.xy * 5.0);
    x.x = _e18.x;
    x.y = _e18.y;
    let _e23 = x;
    let _e27 = (_e23.zy + vec2<f32>(1.0));
    x.z = _e27.x;
    x.y = _e27.y;
    let _e32 = x;
    let _e34 = x;
    local = _e34.xz;
    foo((&local));
    let _e41 = local.x;
    x.x = _e41;
    let _e42 = local.y;
    x.z = _e42;
    return;
}

@stage(fragment) 
fn main() {
    main_1();
    return;
}

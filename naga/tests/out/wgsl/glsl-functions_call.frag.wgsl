fn swizzleCallee(a: ptr<function, vec2<f32>>) {
    return;
}

fn swizzleCaller(a_1: vec3<f32>) {
    var a_2: vec3<f32>;
    var local: vec2<f32>;

    a_2 = a_1;
    let _e2 = a_2;
    local = _e2.xz;
    swizzleCallee((&local));
    let _e9 = local.x;
    a_2.x = _e9;
    let _e10 = local.y;
    a_2.z = _e10;
    return;
}

fn outImplicitCastCallee(a_3: ptr<function, u32>) {
    return;
}

fn outImplicitCastCaller(a_4: f32) {
    var a_5: f32;
    var local_1: u32;

    a_5 = a_4;
    outImplicitCastCallee((&local_1));
    let _e3 = local_1;
    a_5 = f32(_e3);
    return;
}

fn swizzleImplicitCastCallee(a_6: ptr<function, vec2<u32>>) {
    return;
}

fn swizzleImplicitCastCaller(a_7: vec3<f32>) {
    var a_8: vec3<f32>;
    var local_2: vec2<u32>;

    a_8 = a_7;
    swizzleImplicitCastCallee((&local_2));
    let _e7 = local_2.x;
    a_8.x = f32(_e7);
    let _e9 = local_2.y;
    a_8.z = f32(_e9);
    return;
}

fn main_1() {
    var a_9: u32;
    var b: vec2<u32>;

    swizzleCaller(vec3(0f));
    outImplicitCastCallee((&a_9));
    outImplicitCastCaller(1f);
    swizzleImplicitCastCallee((&b));
    swizzleImplicitCastCaller(vec3(0f));
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

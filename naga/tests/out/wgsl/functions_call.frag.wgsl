fn swizzleCallee(a: ptr<function, vec2<f32>>) {
    return;
}

fn swizzleCaller(a_1: vec3<f32>) {
    var a_2: vec3<f32>;
    var local: vec2<f32>;

    a_2 = a_1;
    let _e2 = a_2;
    let _e4 = a_2;
    local = _e4.xz;
    swizzleCallee((&local));
    let _e11 = local.x;
    a_2.x = _e11;
    let _e12 = local.y;
    a_2.z = _e12;
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
    let _e5 = local_1;
    a_5 = f32(_e5);
    return;
}

fn swizzleImplicitCastCallee(a_6: ptr<function, vec2<u32>>) {
    return;
}

fn swizzleImplicitCastCaller(a_7: vec3<f32>) {
    var a_8: vec3<f32>;
    var local_2: vec2<u32>;

    a_8 = a_7;
    let _e2 = a_8;
    let _e4 = a_8;
    swizzleImplicitCastCallee((&local_2));
    let _e11 = local_2.x;
    a_8.x = f32(_e11);
    let _e13 = local_2.y;
    a_8.z = f32(_e13);
    return;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

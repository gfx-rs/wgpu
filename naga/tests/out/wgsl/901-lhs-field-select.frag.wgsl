fn main_1() {
    var a: vec4<f32> = vec4(1f);

    a.x = 2f;
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

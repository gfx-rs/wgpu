fn main_1() {
    var a: vec4<f32> = vec4(1.0);

    a.x = 2.0;
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

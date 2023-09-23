fn main_1() {
    var a: mat4x4<f32>;

    let _e1 = f32(1);
    a = mat4x4<f32>(vec4<f32>(_e1, 0.0, 0.0, 0.0), vec4<f32>(0.0, _e1, 0.0, 0.0), vec4<f32>(0.0, 0.0, _e1, 0.0), vec4<f32>(0.0, 0.0, 0.0, _e1));
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

fn main_1() {
    var a: mat4x4<f32> = mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), vec4<f32>(0.0, 0.0, 0.0, 1.0));

    let _e1 = f32(1);
}

[[stage(vertex)]]
fn main() {
    main_1();
    return;
}

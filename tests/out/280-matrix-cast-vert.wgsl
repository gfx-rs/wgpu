fn main() {
    var a: mat4x4<f32>;

    let _e2: vec4<f32> = vec4<f32>(f32(1));
    a = mat4x4<f32>(_e2, _e2, _e2, _e2);
    return;
}

[[stage(vertex)]]
fn main1() {
    main();
    return;
}

fn main1() {
    var splat: mat2x2<f32>;
    var normal: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 1.0), vec2<f32>(2.0, 2.0));
    var a: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var b: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var c: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0, 2.0, 3.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var d: mat3x3<f32> = mat3x3<f32>(vec3<f32>(2.0, 2.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var e: mat4x4<f32> = mat4x4<f32>(vec4<f32>(2.0, 2.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 2.0, 2.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));

    let _e2: vec2<f32> = vec2<f32>(f32(1));
    splat = mat2x2<f32>(_e2, _e2);
    let _e7: vec2<f32> = vec2<f32>(f32(1));
    let _e10: vec2<f32> = vec2<f32>(f32(2));
    let _e36: vec2<f32> = vec2<f32>(f32(2), f32(3));
    let _e51: vec3<f32> = vec3<f32>(f32(1));
    let _e54: vec3<f32> = vec3<f32>(f32(1));
    let _e71: vec2<f32> = vec2<f32>(f32(2));
    let _e75: vec3<f32> = vec3<f32>(f32(1));
    let _e78: vec3<f32> = vec3<f32>(f32(1));
    let _e95: vec2<f32> = vec2<f32>(f32(2));
    let _e98: vec4<f32> = vec4<f32>(f32(1));
    let _e101: vec2<f32> = vec2<f32>(f32(2));
    let _e104: vec4<f32> = vec4<f32>(f32(1));
    let _e107: vec4<f32> = vec4<f32>(f32(1));
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}

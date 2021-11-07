fn main_1() {
    var splat: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    var normal: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 1.0), vec2<f32>(2.0, 2.0));
    var from_matrix: mat2x4<f32> = mat2x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), vec4<f32>(0.0, 1.0, 0.0, 0.0));
    var a: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var b: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var c: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0, 2.0, 3.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var d: mat3x3<f32> = mat3x3<f32>(vec3<f32>(2.0, 2.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var e: mat4x4<f32> = mat4x4<f32>(vec4<f32>(2.0, 2.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 2.0, 2.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));

    let e1: f32 = f32(1);
    let e9: vec2<f32> = vec2<f32>(f32(1));
    let e12: vec2<f32> = vec2<f32>(f32(2));
    let e26: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
    let e58: vec2<f32> = vec2<f32>(f32(2), f32(3));
    let e73: vec3<f32> = vec3<f32>(f32(1));
    let e76: vec3<f32> = vec3<f32>(f32(1));
    let e93: vec2<f32> = vec2<f32>(f32(2));
    let e97: vec3<f32> = vec3<f32>(f32(1));
    let e100: vec3<f32> = vec3<f32>(f32(1));
    let e117: vec2<f32> = vec2<f32>(f32(2));
    let e120: vec4<f32> = vec4<f32>(f32(1));
    let e123: vec2<f32> = vec2<f32>(f32(2));
    let e126: vec4<f32> = vec4<f32>(f32(1));
    let e129: vec4<f32> = vec4<f32>(f32(1));
}

[[stage(vertex)]]
fn main() {
    main_1();
    return;
}

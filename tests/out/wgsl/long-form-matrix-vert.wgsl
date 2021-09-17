fn main1() {
    var splat: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    var normal: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 1.0), vec2<f32>(2.0, 2.0));
    var a: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var b: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var c: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0, 2.0, 3.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var d: mat3x3<f32> = mat3x3<f32>(vec3<f32>(2.0, 2.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var e: mat4x4<f32> = mat4x4<f32>(vec4<f32>(2.0, 2.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 2.0, 2.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));

    let e1: f32 = f32(1);
    let e9: vec2<f32> = vec2<f32>(f32(1));
    let e12: vec2<f32> = vec2<f32>(f32(2));
    let e38: vec2<f32> = vec2<f32>(f32(2), f32(3));
    let e53: vec3<f32> = vec3<f32>(f32(1));
    let e56: vec3<f32> = vec3<f32>(f32(1));
    let e73: vec2<f32> = vec2<f32>(f32(2));
    let e77: vec3<f32> = vec3<f32>(f32(1));
    let e80: vec3<f32> = vec3<f32>(f32(1));
    let e97: vec2<f32> = vec2<f32>(f32(2));
    let e100: vec4<f32> = vec4<f32>(f32(1));
    let e103: vec2<f32> = vec2<f32>(f32(2));
    let e106: vec4<f32> = vec4<f32>(f32(1));
    let e109: vec4<f32> = vec4<f32>(f32(1));
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}

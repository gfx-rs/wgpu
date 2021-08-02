fn main1() {
    var splat: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0));
    var normal: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 1.0), vec2<f32>(2.0, 2.0));
    var a: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var b: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1.0, 2.0), vec2<f32>(3.0, 4.0));
    var c: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1.0, 2.0, 3.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var d: mat3x3<f32> = mat3x3<f32>(vec3<f32>(2.0, 2.0, 1.0), vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0));
    var e: mat4x4<f32> = mat4x4<f32>(vec4<f32>(2.0, 2.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 2.0, 2.0), vec4<f32>(1.0, 1.0, 1.0, 1.0), vec4<f32>(1.0, 1.0, 1.0, 1.0));

    let _e1: f32 = f32(1);
    let _e9: vec2<f32> = vec2<f32>(f32(1));
    let _e12: vec2<f32> = vec2<f32>(f32(2));
    let _e38: vec2<f32> = vec2<f32>(f32(2), f32(3));
    let _e53: vec3<f32> = vec3<f32>(f32(1));
    let _e56: vec3<f32> = vec3<f32>(f32(1));
    let _e73: vec2<f32> = vec2<f32>(f32(2));
    let _e77: vec3<f32> = vec3<f32>(f32(1));
    let _e80: vec3<f32> = vec3<f32>(f32(1));
    let _e97: vec2<f32> = vec2<f32>(f32(2));
    let _e100: vec4<f32> = vec4<f32>(f32(1));
    let _e103: vec2<f32> = vec2<f32>(f32(2));
    let _e106: vec4<f32> = vec4<f32>(f32(1));
    let _e109: vec4<f32> = vec4<f32>(f32(1));
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}

fn main_1() {
    var splat: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 0f), vec2<f32>(0f, 1f));
    var normal: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 1f), vec2<f32>(2f, 2f));
    var from_matrix: mat2x4<f32> = mat2x4<f32>(vec4<f32>(1f, 0f, 0f, 0f), vec4<f32>(0f, 1f, 0f, 0f));
    var a: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f));
    var b: mat2x2<f32> = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f));
    var c: mat3x3<f32> = mat3x3<f32>(vec3<f32>(1f, 2f, 3f), vec3<f32>(1f, 1f, 1f), vec3<f32>(1f, 1f, 1f));
    var d: mat3x3<f32> = mat3x3<f32>(vec3<f32>(2f, 2f, 1f), vec3<f32>(1f, 1f, 1f), vec3<f32>(1f, 1f, 1f));
    var e: mat4x4<f32> = mat4x4<f32>(vec4<f32>(2f, 2f, 1f, 1f), vec4<f32>(1f, 1f, 2f, 2f), vec4<f32>(1f, 1f, 1f, 1f), vec4<f32>(1f, 1f, 1f, 1f));

}

@fragment 
fn main() {
    main_1();
    return;
}

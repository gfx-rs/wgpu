@compute @workgroup_size(1, 1, 1) 
fn main() {
    let a = vec3<f32>(0f, 0f, 0f);
    let c = vec3(0f);
    let b = vec3<f32>(vec2(0f), 0f);
    let d = vec3<f32>(vec2(0f), 0f);
    let e = vec3<i32>(d);
    let f = mat2x2<f32>(vec2<f32>(1f, 2f), vec2<f32>(3f, 4f));
    let g = mat3x3<f32>(a, a, a);
    return;
}

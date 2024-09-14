@compute @workgroup_size(1, 1, 1) 
fn main() {
    let a = cross(vec3<f32>(0f, 1f, 2f), vec3<f32>(0f, 1f, 2f));
}

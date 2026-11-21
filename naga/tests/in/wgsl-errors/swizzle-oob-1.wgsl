// 3-component swizzle from const vec2

@compute @workgroup_size(1)
fn main() {
    const v = vec2<i32>();
    let r : vec3<i32> = v.xyz;
}

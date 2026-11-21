// 4-component swizzle from non-const vec3

@compute @workgroup_size(1)
fn main() {
    var v = vec3<i32>();
    let r : vec4<i32> = v.xyzw;
}

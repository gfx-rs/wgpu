// Split up because some output languages limit number of locations to 8.
struct FragmentOutputVec4Vec3 {
    @location(0) vec4f: vec4<f32>,
    @location(1) vec4i: vec4<i32>,
    @location(2) vec4u: vec4<u32>,
    @location(3) vec3f: vec3<f32>,
    @location(4) vec3i: vec3<i32>,
    @location(5) vec3u: vec3<u32>,
}
@fragment
fn main_vec4vec3() -> FragmentOutputVec4Vec3 {
    var output: FragmentOutputVec4Vec3;
    output.vec4f = vec4<f32>(0.0);
    output.vec4i = vec4<i32>(0);
    output.vec4u = vec4<u32>(0u);
    output.vec3f = vec3<f32>(0.0);
    output.vec3i = vec3<i32>(0);
    output.vec3u = vec3<u32>(0u);
    return output;
}

struct FragmentOutputVec2Scalar {
    @location(0) vec2f: vec2<f32>,
    @location(1) vec2i: vec2<i32>,
    @location(2) vec2u: vec2<u32>,
    @location(3) scalarf: f32,
    @location(4) scalari: i32,
    @location(5) scalaru: u32,
}

@fragment
fn main_vec2scalar() -> FragmentOutputVec2Scalar {
    var output: FragmentOutputVec2Scalar;
    output.vec2f = vec2<f32>(0.0);
    output.vec2i = vec2<i32>(0);
    output.vec2u = vec2<u32>(0u);
    output.scalarf = 0.0;
    output.scalari = 0;
    output.scalaru = 0u;
    return output;
}

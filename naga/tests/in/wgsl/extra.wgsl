enable primitive_index;

struct ImmediateData {
    index: u32,
    double: vec2<f32>,
}
var<immediate> im: ImmediateData;

struct FragmentIn {
    @location(0) color: vec4<f32>,
    @builtin(primitive_index) primitive_index: u32,
}

@fragment
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    if in.primitive_index == im.index {
        return in.color;
    } else {
        return vec4<f32>(vec3<f32>(1.0) - in.color.rgb, in.color.a);
    }
}

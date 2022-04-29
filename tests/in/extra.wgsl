struct PushConstants {
    index: u32,
    double: vec2<f64>,
}
var<push_constant> pc: PushConstants;

struct FragmentIn {
    @location(0) color: vec4<f32>,
    @builtin(primitive_index) primitive_index: u32,
}

@fragment
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    if in.primitive_index == pc.index {
        return in.color;
    } else {
        return vec4<f32>(vec3<f32>(1.0) - in.color.rgb, in.color.a);
    }
}

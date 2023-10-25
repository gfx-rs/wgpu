struct PushConstants {
    index: u32,
    double: vec2<f64>,
}

struct FragmentIn {
    @location(0) color: vec4<f32>,
    @builtin(primitive_index) primitive_index: u32,
}

var<push_constant> pc: PushConstants;

@fragment 
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    let _e4 = pc.index;
    if (in.primitive_index == _e4) {
        return in.color;
    } else {
        return vec4<f32>((vec3(1.0) - in.color.xyz), in.color.w);
    }
}

struct PushConstants {
    multiplier: f32
}
var<push_constant> pc: PushConstants;

struct FragmentIn {
    @location(0) color: vec4<f32>
}

@fragment
fn main(_in: FragmentIn) -> @location(0) vec4<f32> {
    return _in.color * pc.multiplier;
}

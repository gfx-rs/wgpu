struct PushConstants {
    index: u32,
    double: vec2<f32>,
}

struct FragmentIn {
    @location(0) color: vec4<f32>,
    @builtin(primitive_index) primitive_index: u32,
}

var<push_constant> pc: PushConstants;

@fragment 
fn main(_in: FragmentIn) -> @location(0) vec4<f32> {
    let _e4 = pc.index;
    if (_in.primitive_index == _e4) {
        return _in.color;
    } else {
        return vec4<f32>((vec3<f32>(1.0) - _in.color.xyz), _in.color.w);
    }
}

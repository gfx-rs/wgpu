[[block]]
struct PushConstants {
    index: u32;
    double: vec2<f32>;
};

var<push_constant> pc: PushConstants;

[[stage(fragment)]]
fn main([[location(0)]] color: vec4<f32>) -> [[location(0)]] vec4<f32> {
    return color;
}

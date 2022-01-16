struct PushConstants {
    multiplier: f32;
};
var<push_constant> pc: PushConstants;

struct FragmentIn {
    [[location(0)]] color: vec4<f32>;
};

[[stage(fragment)]]
fn main(in: FragmentIn) -> [[location(0)]] vec4<f32> {
    return in.color * pc.multiplier;
}

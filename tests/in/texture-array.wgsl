[[group(0), binding(0)]] var texture0: texture_2d<f32>;
[[group(0), binding(1)]] var texture1: texture_2d<f32>;
[[group(0), binding(2)]] var sampler: sampler;

[[block]]
struct PushConstants {
    index: u32;
};
var<push_constant> pc: PushConstants;

[[stage(fragment)]]
fn main([[location(0)]] tex_coord: vec2<f32>) -> [[location(1)]] vec4<f32> {
    if (pc.index == 0u) {
        return textureSample(texture0, sampler, tex_coord);
    } else {
        return textureSample(texture1, sampler, tex_coord);
    }
}

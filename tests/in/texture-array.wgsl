[[location(0)]] var<in> tex_coord: vec2<f32>;
[[group(0), binding(0)]] var texture0: texture_2d<f32>;
[[group(0), binding(1)]] var texture1: texture_2d<f32>;
[[group(0), binding(2)]] var sampler: sampler;

[[block]]
struct PushConstants {
    index: u32;
};
var<push_constant> pc: PushConstants;

[[location(1)]] var<out> color: vec4<f32>;

[[stage(fragment)]]
fn main() {
    if (pc.index == 0) {
        color = textureSample(texture0, sampler, tex_coord);
    } else {
        color = textureSample(texture1, sampler, tex_coord);
    }
}

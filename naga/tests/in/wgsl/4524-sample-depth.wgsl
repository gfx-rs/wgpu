@group(0) @binding(0)
var tex: texture_depth_2d;

@group(0) @binding(1)
var samp: sampler;

@fragment
fn main() {
    let a = textureSample(tex, samp, vec2f(0.0));
}
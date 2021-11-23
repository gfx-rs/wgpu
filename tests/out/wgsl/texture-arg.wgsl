[[group(0), binding(0)]]
var Texture: texture_2d<f32>;
[[group(0), binding(1)]]
var Sampler: sampler;

fn test(Passed_Texture: texture_2d<f32>, Passed_Sampler: sampler) -> vec4<f32> {
    let _e7: vec4<f32> = textureSample(Passed_Texture, Passed_Sampler, vec2<f32>(0.0, 0.0));
    return _e7;
}

[[stage(fragment)]]
fn main() -> [[location(0)]] vec4<f32> {
    let _e2: vec4<f32> = test(Texture, Sampler);
    return _e2;
}

@group(0) @binding(0) var sampler1 : sampler;
@group(0) @binding(1) var texture : texture_2d<bool>;

@fragment
fn main() -> @location(0) vec4<f32> {
    return textureSample(texture, sampler1, vec2<f32>(0.0));
}

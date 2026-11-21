@group(0) @binding(0) var sampler1 : sampler;

@fragment
fn main() -> @location(0) vec4<f32> {
    let a = 3;
    return textureSample(a, sampler1, vec2<f32>(0.0));
}

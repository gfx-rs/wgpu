enable wgpu_binding_array;

@group(0) @binding(0)
var textures: binding_array<texture_2d<f32>, 4>;
@group(0) @binding(1)
var samplers: binding_array<sampler, 4>;

fn sample(t: binding_array<texture_2d<f32>, 4>, s: binding_array<sampler, 4>, i: u32, uv: vec2<f32>) -> vec4<f32> {
    return textureSample(t[i], s[i], uv);
}

fn sample_forward(t: binding_array<texture_2d<f32>, 4>, s: binding_array<sampler, 4>, uv: vec2<f32>) -> vec4<f32> {
    return sample(t, s, 1u, uv) + sample(t, s, 2u, uv);
}

fn size(t: binding_array<texture_2d<f32>, 4>, i: u32) -> vec2<u32> {
    return textureDimensions(t[i]);
}

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let v = sample(textures, samplers, 0u, uv) + sample_forward(textures, samplers, uv);
    return v * f32(size(textures, 3u).x);
}

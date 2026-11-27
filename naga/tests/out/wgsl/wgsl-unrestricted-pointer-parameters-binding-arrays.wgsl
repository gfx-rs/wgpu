enable wgpu_binding_array;

@group(0) @binding(0)
var textures: binding_array<texture_2d<f32>, 4>;
@group(0) @binding(1)
var samplers: binding_array<sampler, 4>;

fn sample(t: binding_array<texture_2d<f32>, 4>, s: binding_array<sampler, 4>, i: u32, uv_1: vec2<f32>) -> vec4<f32> {
    let _e6 = textureSample(t[i], s[i], uv_1);
    return _e6;
}

fn sample_forward(t_1: binding_array<texture_2d<f32>, 4>, s_1: binding_array<sampler, 4>, uv_2: vec2<f32>) -> vec4<f32> {
    let _e4 = sample(t_1, s_1, 1u, uv_2);
    let _e6 = sample(t_1, s_1, 2u, uv_2);
    return (_e4 + _e6);
}

fn size(t_2: binding_array<texture_2d<f32>, 4>, i_1: u32) -> vec2<u32> {
    let _e3 = textureDimensions(t_2[i_1]);
    return _e3;
}

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let _e4 = sample(textures, samplers, 0u, uv);
    let _e7 = sample_forward(textures, samplers, uv);
    let v = (_e4 + _e7);
    let _e11 = size(textures, 3u);
    return (v * f32(_e11.x));
}

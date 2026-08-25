@group(0) @binding(0)
var sampled_texture_1d: texture_1d<f32>;

@group(0) @binding(1)
var sampled_texture_2d: texture_2d<f32>;

@group(0) @binding(2)
var sampled_texture_3d: texture_3d<f32>;

@group(0) @binding(3)
var texture_sampler: sampler;

@fragment
fn main() -> @location(0) vec4<f32> {
    let sample_1d = textureSampleLevel(sampled_texture_1d, texture_sampler, 0.5, 0.0, -1);
    let sample_2d = textureSampleLevel(
        sampled_texture_2d,
        texture_sampler,
        vec2<f32>(0.5),
        0.0,
        vec2<i32>(-1, 2),
    );
    let sample_3d = textureSampleLevel(
        sampled_texture_3d,
        texture_sampler,
        vec3<f32>(0.5),
        0.0,
        vec3<i32>(-1, 2, -3),
    );
    return sample_1d + sample_2d + sample_3d;
}

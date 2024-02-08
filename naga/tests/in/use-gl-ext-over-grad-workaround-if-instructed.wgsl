// see https://github.com/gfx-rs/wgpu/pull/5171

@group(0) @binding(0) var texture: texture_depth_2d_array;
@group(0) @binding(1) var texture_sampler: sampler_comparison;

@fragment
fn main() -> @location(0) f32  {
    let pos = vec2<f32>(0.0);
    let array_index: i32 = 0;
    let depth: f32 = 0.0;
    return textureSampleCompareLevel(texture, texture_sampler, pos, array_index, depth);
}

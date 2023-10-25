struct UniformIndex {
    index: u32
};

@group(0) @binding(0)
var texture_array_unbounded: binding_array<texture_2d<f32>>;
@group(0) @binding(1)
var texture_array_bounded: binding_array<texture_2d<f32>, 5>;
@group(0) @binding(2)
var texture_array_2darray: binding_array<texture_2d_array<f32>, 5>;
@group(0) @binding(3)
var texture_array_multisampled: binding_array<texture_multisampled_2d<f32>, 5>;
@group(0) @binding(4)
var texture_array_depth: binding_array<texture_depth_2d, 5>;
@group(0) @binding(5)
var texture_array_storage: binding_array<texture_storage_2d<rgba32float, write>, 5>;
@group(0) @binding(6)
var samp: binding_array<sampler, 5>;
@group(0) @binding(7)
var samp_comp: binding_array<sampler_comparison, 5>;
@group(0) @binding(8)
var<uniform> uni: UniformIndex;

struct FragmentIn {
    @location(0) index: u32,
};

@fragment
fn main(fragment_in: FragmentIn) -> @location(0) vec4<f32> {
    let uniform_index = uni.index;
    let non_uniform_index = fragment_in.index;

    var u1 = 0u;
    var u2 = vec2<u32>(0u);
    var v1 = 0.0;
    var v4 = vec4<f32>(0.0);
    
    // This example is arranged in the order of the texture definitions in the wgsl spec
    // 
    // The first function uses texture_array_unbounded, the rest use texture_array_bounded to make sure
    // they both show up in the output. Functions that need depth use texture_array_2darray.
    //
    // We only test 2D f32 textures here as the machinery for binding indexing doesn't care about
    // texture format or texture dimension.

    let uv = vec2<f32>(0.0);
    let pix = vec2<i32>(0);

    u2 += textureDimensions(texture_array_unbounded[0]);
    u2 += textureDimensions(texture_array_unbounded[uniform_index]);
    u2 += textureDimensions(texture_array_unbounded[non_uniform_index]);

    v4 += textureGather(0, texture_array_bounded[0], samp[0], uv);
    v4 += textureGather(0, texture_array_bounded[uniform_index], samp[uniform_index], uv);
    v4 += textureGather(0, texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv); 

    v4 += textureGatherCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    v4 += textureGatherCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    v4 += textureGatherCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0); 

    v4 += textureLoad(texture_array_unbounded[0], pix, 0);
    v4 += textureLoad(texture_array_unbounded[uniform_index], pix, 0);
    v4 += textureLoad(texture_array_unbounded[non_uniform_index], pix, 0);

    u1 += textureNumLayers(texture_array_2darray[0]);
    u1 += textureNumLayers(texture_array_2darray[uniform_index]);
    u1 += textureNumLayers(texture_array_2darray[non_uniform_index]);

    u1 += textureNumLevels(texture_array_bounded[0]);
    u1 += textureNumLevels(texture_array_bounded[uniform_index]);
    u1 += textureNumLevels(texture_array_bounded[non_uniform_index]);

    u1 += textureNumSamples(texture_array_multisampled[0]);
    u1 += textureNumSamples(texture_array_multisampled[uniform_index]);
    u1 += textureNumSamples(texture_array_multisampled[non_uniform_index]);

    v4 += textureSample(texture_array_bounded[0], samp[0], uv);
    v4 += textureSample(texture_array_bounded[uniform_index], samp[uniform_index], uv);
    v4 += textureSample(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv);

    v4 += textureSampleBias(texture_array_bounded[0], samp[0], uv, 0.0);
    v4 += textureSampleBias(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    v4 += textureSampleBias(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);

    v1 += textureSampleCompare(texture_array_depth[0], samp_comp[0], uv, 0.0);
    v1 += textureSampleCompare(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    v1 += textureSampleCompare(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);

    v1 += textureSampleCompareLevel(texture_array_depth[0], samp_comp[0], uv, 0.0);
    v1 += textureSampleCompareLevel(texture_array_depth[uniform_index], samp_comp[uniform_index], uv, 0.0);
    v1 += textureSampleCompareLevel(texture_array_depth[non_uniform_index], samp_comp[non_uniform_index], uv, 0.0);

    v4 += textureSampleGrad(texture_array_bounded[0], samp[0], uv, uv, uv);
    v4 += textureSampleGrad(texture_array_bounded[uniform_index], samp[uniform_index], uv, uv, uv);
    v4 += textureSampleGrad(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, uv, uv);

    v4 += textureSampleLevel(texture_array_bounded[0], samp[0], uv, 0.0);
    v4 += textureSampleLevel(texture_array_bounded[uniform_index], samp[uniform_index], uv, 0.0);
    v4 += textureSampleLevel(texture_array_bounded[non_uniform_index], samp[non_uniform_index], uv, 0.0);

    textureStore(texture_array_storage[0], pix, v4);
    textureStore(texture_array_storage[uniform_index], pix, v4);
    textureStore(texture_array_storage[non_uniform_index], pix, v4);

    let v2 = vec2<f32>(u2 + vec2<u32>(u1));

    return v4 + vec4<f32>(v2.x, v2.y, v2.x, v2.y) + v1;
}

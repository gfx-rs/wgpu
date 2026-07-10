// Exercises the HLSL backend's `SamplerBinding::Direct` option, which declares
// samplers as ordinary HLSL `SamplerState`/`SamplerComparisonState` bound to a
// register, rather than routing them through the D3D12 sampler heap.
enable wgpu_binding_array;

@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var samp_comp: sampler_comparison;
@group(0) @binding(4) var samp_array: binding_array<sampler, 4>;

struct FragmentIn {
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) index: u32,
};

@fragment
fn main(in: FragmentIn) -> @location(0) vec4<f32> {
    var color = vec4<f32>(0.0);

    // A plain sampler: `SamplerState`.
    color += textureSample(tex, samp, in.uv);

    // A comparison sampler: `SamplerComparisonState`.
    color.x += textureSampleCompare(depth_tex, samp_comp, in.uv, 0.5);

    // A sampler binding array, indexed by a constant and a (non-uniform) value.
    color += textureSample(tex, samp_array[0], in.uv);
    color += textureSample(tex, samp_array[in.index], in.uv);

    return color;
}

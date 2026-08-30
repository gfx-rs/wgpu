enable resource_table;

@group(0) @binding(0) var samp: sampler;

// Only reached from `fs_main`, and only indirectly: exercises the "a
// `getResource` may live in a called helper" case from
// `write_resource_table_globals`'s doc comment (naga/src/back/spv/writer.rs).
fn sample_table(index: u32, uv: vec2<f32>) -> vec4<f32> {
    let tex = getResource<texture_2d<f32>>(index);
    return textureSample(tex, samp, uv);
}

// Does NOT use the resource table at all. Its `OpEntryPoint` interface
// should not *need* the synthesized table global(s), but the SPIR-V backend
// over-lists them into every entry point's interface (a safe superset per
// spirv-val) rather than tracking per-entry-point reachability.
@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn fs_main(
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) index: u32,
) -> @location(0) vec4<f32> {
    return sample_table(index, pos.xy);
}

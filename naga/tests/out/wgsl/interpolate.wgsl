struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) _flat: u32,
    @location(1) @interpolate(flat, first) flat_first: u32,
    @location(2) @interpolate(flat, either) flat_either: u32,
    @location(3) @interpolate(linear) _linear: f32,
    @location(4) @interpolate(linear, centroid) linear_centroid: vec2<f32>,
    @location(6) @interpolate(linear, sample) linear_sample: vec3<f32>,
    @location(7) perspective: vec4<f32>,
    @location(8) @interpolate(perspective, centroid) perspective_centroid: f32,
    @location(9) @interpolate(perspective, sample) perspective_sample: f32,
}

@vertex 
fn vert_main() -> FragmentInput {
    var out: FragmentInput;

    out.position = vec4<f32>(2f, 4f, 5f, 6f);
    out._flat = 8u;
    out.flat_first = 9u;
    out.flat_either = 10u;
    out._linear = 27f;
    out.linear_centroid = vec2<f32>(64f, 125f);
    out.linear_sample = vec3<f32>(216f, 343f, 512f);
    out.perspective = vec4<f32>(729f, 1000f, 1331f, 1728f);
    out.perspective_centroid = 2197f;
    out.perspective_sample = 2744f;
    let _e34 = out;
    return _e34;
}

@fragment 
fn frag_main(val: FragmentInput) {
    return;
}

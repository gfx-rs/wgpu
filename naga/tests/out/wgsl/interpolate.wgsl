struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) _flat: u32,
    @location(1) @interpolate(flat, first) flat_first: u32,
    @location(2) @interpolate(flat, either) flat_either: u32,
    @location(3) @interpolate(linear) _linear: f32,
    @location(4) @interpolate(linear, centroid) linear_centroid: vec2<f32>,
    @location(6) @interpolate(linear, sample) linear_sample: vec3<f32>,
    @location(7) @interpolate(linear) linear_center: vec3<f32>,
    @location(8) perspective: vec4<f32>,
    @location(9) @interpolate(perspective, centroid) perspective_centroid: f32,
    @location(10) @interpolate(perspective, sample) perspective_sample: f32,
    @location(11) perspective_center: f32,
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
    out.linear_center = vec3<f32>(255f, 511f, 1024f);
    out.perspective = vec4<f32>(729f, 1000f, 1331f, 1728f);
    out.perspective_centroid = 2197f;
    out.perspective_sample = 2744f;
    out.perspective_center = 2812f;
    let _e41 = out;
    return _e41;
}

@fragment 
fn frag_main(val: FragmentInput) {
    return;
}

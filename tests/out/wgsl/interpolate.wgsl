struct FragmentInput {
    @builtin(position) position: vec4<f32>,
    @location(0) _flat: u32,
    @location(1) @interpolate(linear) _linear: f32,
    @location(2) @interpolate(linear, centroid) linear_centroid: vec2<f32>,
    @location(3) @interpolate(linear, sample) linear_sample: vec3<f32>,
    @location(4) perspective: vec4<f32>,
    @location(5) @interpolate(perspective, centroid) perspective_centroid: f32,
    @location(6) @interpolate(perspective, sample) perspective_sample: f32,
}

@vertex 
fn vert_main() -> FragmentInput {
    var _out: FragmentInput;

    _out.position = vec4<f32>(2.0, 4.0, 5.0, 6.0);
    _out._flat = 8u;
    _out._linear = 27.0;
    _out.linear_centroid = vec2<f32>(64.0, 125.0);
    _out.linear_sample = vec3<f32>(216.0, 343.0, 512.0);
    _out.perspective = vec4<f32>(729.0, 1000.0, 1331.0, 1728.0);
    _out.perspective_centroid = 2197.0;
    _out.perspective_sample = 2744.0;
    let _e30 = _out;
    return _e30;
}

@fragment 
fn frag_main(val: FragmentInput) {
    return;
}

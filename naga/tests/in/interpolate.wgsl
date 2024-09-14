//TODO: merge with "interface"?

// NOTE: invalid combinations are tested in the
// `validation::incompatible_interpolation_and_sampling_types` test.
struct FragmentInput {
  @builtin(position) position: vec4<f32>,
  @location(0) @interpolate(flat) _flat : u32,
  @location(1) @interpolate(flat, first) flat_first : u32,
  @location(2) @interpolate(flat, either) flat_either : u32,
  @location(3) @interpolate(linear) _linear : f32,
  @location(4) @interpolate(linear, centroid) linear_centroid : vec2<f32>,
  @location(6) @interpolate(linear, sample) linear_sample : vec3<f32>,
  @location(7) @interpolate(linear, center) linear_center : vec3<f32>,
  @location(8) @interpolate(perspective) perspective : vec4<f32>,
  @location(9) @interpolate(perspective, centroid) perspective_centroid : f32,
  @location(10) @interpolate(perspective, sample) perspective_sample : f32,
  @location(11) @interpolate(perspective, center) perspective_center : f32,
}

@vertex
fn vert_main() -> FragmentInput {
   var out: FragmentInput;

   out.position = vec4<f32>(2.0, 4.0, 5.0, 6.0);
   out._flat = 8u;
   out.flat_first = 9u;
   out.flat_either = 10u;
   out._linear = 27.0;
   out.linear_centroid = vec2<f32>(64.0, 125.0);
   out.linear_sample = vec3<f32>(216.0, 343.0, 512.0);
   out.linear_center = vec3<f32>(255.0, 511.0, 1024.0);
   out.perspective = vec4<f32>(729.0, 1000.0, 1331.0, 1728.0);
   out.perspective_centroid = 2197.0;
   out.perspective_sample = 2744.0;
   out.perspective_center = 2812.0;

   return out;
}

@fragment
fn frag_main(val : FragmentInput) { }

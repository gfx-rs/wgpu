struct FragmentInput {
  [[builtin(position)]] position: vec4<f32>;
  [[location(0), interpolate(flat)]] flat : u32;
  [[location(1), interpolate(linear)]] linear : f32;
  [[location(2), interpolate(linear,centroid)]] linear_centroid : vec2<f32>;
  [[location(3), interpolate(linear,sample)]] linear_sample : vec3<f32>;
  [[location(4), interpolate(perspective)]] perspective : vec4<f32>;
  [[location(5), interpolate(perspective,centroid)]] perspective_centroid : f32;
  [[location(6), interpolate(perspective,sample)]] perspective_sample : f32;
};

[[stage(vertex)]]
fn main() -> FragmentInput {
   var out: FragmentInput;

   out.position = vec4<f32>(2.0, 4.0, 5.0, 6.0);
   out.flat = 8u32;
   out.linear = 27.0;
   out.linear_centroid = vec2<f32>(64.0, 125.0);
   out.linear_sample = vec3<f32>(216.0, 343.0, 512.0);
   out.perspective = vec4<f32>(729.0, 1000.0, 1331.0, 1728.0);
   out.perspective_centroid = 2197.0;
   out.perspective_sample = 2744.0;

   return out;
}

[[stage(fragment)]]
fn main(val : FragmentInput) { }

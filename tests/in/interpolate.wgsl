struct FragmentInput {
  [[builtin(position)]] position: vec4<f32>;
  [[location(0), interpolate(flat)]] flat : u32;
  [[location(1), interpolate(linear)]] linear: f32;
  [[location(2), interpolate(centroid)]] centroid: vec2<f32>;
  [[location(3), interpolate(sample)]] sample: vec3<f32>;
  [[location(4), interpolate(perspective)]] perspective: vec4<f32>;
};

[[stage(vertex)]]
fn main() -> FragmentInput {
   var out: FragmentInput;

   out.position = vec4<f32>(2.0, 4.0, 5.0, 6.0);
   out.flat = 8u32;
   out.linear = 27.0;
   out.centroid = vec2<f32>(64.0, 125.0);
   out.sample = vec3<f32>(216.0, 343.0, 512.0);
   out.perspective = vec4<f32>(729.0, 1000.0, 1331.0, 1728.0);

   return out;
}

[[stage(fragment)]]
fn main(val : FragmentInput) { }

override o: f32;

@group(0) @binding(0)
var acc_struct: acceleration_structure;

@compute @workgroup_size(1)
fn main() {
  var rq: ray_query;

  let desc = RayDesc(
      RAY_FLAG_TERMINATE_ON_FIRST_HIT,
      0xFFu,
      o * 17.0,
      o * 19.0,
      vec3<f32>(o * 23.0),
      vec3<f32>(o * 29.0, o * 31.0, o * 37.0),
  );
  rayQueryInitialize(&rq, acc_struct, desc);

  while (rayQueryProceed(&rq)) {}
}

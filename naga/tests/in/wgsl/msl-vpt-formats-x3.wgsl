// Tests for the vertex pulling transform in the MSL backend.
//
// Test loading `vec3` from each vertex format.
//
// (Note: only the wgsl files are different for each of the -formats-xN tests.
// The toml files are all the same.)

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
}

struct VertexInput {
  @location( 0) v_uint8     : vec3<u32>,
  @location( 1) v_uint8x2   : vec3<u32>,
  @location( 2) v_uint8x4   : vec3<u32>,
  @location( 3) v_sint8     : vec3<i32>,
  @location( 4) v_sint8x2   : vec3<i32>,
  @location( 5) v_sint8x4   : vec3<i32>,
  @location( 6) v_unorm8    : vec3<f32>,
  @location( 7) v_unorm8x2  : vec3<f32>,
  @location( 8) v_unorm8x4  : vec3<f32>,
  @location( 9) v_snorm8    : vec3<f32>,
  @location(10) v_snorm8x2  : vec3<f32>,
  @location(11) v_snorm8x4  : vec3<f32>,
  @location(12) v_uint16    : vec3<u32>,
  @location(13) v_uint16x2  : vec3<u32>,
  @location(14) v_uint16x4  : vec3<u32>,
  @location(15) v_sint16    : vec3<i32>,
  @location(16) v_sint16x2  : vec3<i32>,
  @location(17) v_sint16x4  : vec3<i32>,
  @location(18) v_unorm16   : vec3<f32>,
  @location(19) v_unorm16x2 : vec3<f32>,
  @location(20) v_unorm16x4 : vec3<f32>,
  @location(21) v_snorm16   : vec3<f32>,
  @location(22) v_snorm16x2 : vec3<f32>,
  @location(23) v_snorm16x4 : vec3<f32>,
  @location(24) v_float16   : vec3<f32>,
  @location(25) v_float16x2 : vec3<f32>,
  @location(26) v_float16x4 : vec3<f32>,
  @location(27) v_float32   : vec3<f32>,
  @location(28) v_float32x2 : vec3<f32>,
  @location(29) v_float32x3 : vec3<f32>,
  @location(30) v_float32x4 : vec3<f32>,
  @location(31) v_uint32    : vec3<u32>,
  @location(32) v_uint32x2  : vec3<u32>,
  @location(33) v_uint32x3  : vec3<u32>,
  @location(34) v_uint32x4  : vec3<u32>,
  @location(35) v_sint32    : vec3<i32>,
  @location(36) v_sint32x2  : vec3<i32>,
  @location(37) v_sint32x3  : vec3<i32>,
  @location(38) v_sint32x4  : vec3<i32>,
  @location(39) v_unorm10_10_10_2: vec3<f32>,
  @location(40) v_unorm8x4_bgra: vec3<f32>,
}

@vertex
fn render_vertex(
  v_in: VertexInput,
) -> VertexOutput
{
  return VertexOutput(vec4f(v_in.v_uint8.x));
}

@group(0) @binding(0)
var<storage, read_write> checksums: array<f32>;

const index_uint = 0u;
const index_sint = 1u;
const index_unorm = 2u;
const index_snorm = 3u;
const index_float16 = 4u;
const index_float32 = 5u;

fn init_checksums() {
  checksums[index_uint] = 0.0;
  checksums[index_sint] = 0.0;
  checksums[index_unorm] = 0.0;
  checksums[index_snorm] = 0.0;
  checksums[index_float16] = 0.0;
  checksums[index_float32] = 0.0;
}

// Break down the 31 vertex formats specified at
// https://gpuweb.github.io/gpuweb/#vertex-formats into blocks
// of 8, to keep under the limits of max locations. Each
// AttributeBlockX structure will get a corresponding
// vertex_block_X function to process its attributes into
// values written to the checksums buffer.

struct AttributeBlock0 {
  // 4-byte-aligned unorm formats
  @location(0) unorm8x4: vec4<f32>,
  @location(1) unorm16x2: vec2<f32>,
  @location(2) unorm16x4: vec4<f32>,

  // 4-byte-aligned snorm formats
  @location(3) snorm8x4: vec4<f32>,
  @location(4) snorm16x2: vec2<f32>,
  @location(5) snorm16x4: vec4<f32>,

  // 2-byte-aligned formats
  @location(6) unorm8x2: vec2<f32>,
  @location(7) snorm8x2: vec2<f32>,
}

@vertex
fn vertex_block_0(v_in: AttributeBlock0) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all unorm into one checksum value.
  var all_unorm: f32 = 0.0;
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x2.x);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x2.y);

  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4.x);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4.y);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4.z);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4.w);

  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16x2.x);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16x2.y);

  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16x4.x);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16x4.y);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16x4.z);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16x4.w);

  checksums[index_unorm] = f32(all_unorm);

  // Accumulate all snorm into one checksum value.
  var all_snorm: f32 = 0.0;
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8x2.x);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8x2.y);

  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8x4.x);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8x4.y);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8x4.z);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8x4.w);

  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16x2.x);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16x2.y);

  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16x4.x);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16x4.y);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16x4.z);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16x4.w);

  checksums[index_snorm] = f32(all_snorm);

  return vec4(0.0);
}

struct AttributeBlock1 {
  // 4-byte-aligned uint formats
  @location(0) uint8x4: vec4<u32>,
  @location(1) uint16x2: vec2<u32>,
  @location(2) uint16x4: vec4<u32>,

  // 4-byte-aligned sint formats
  @location(3) sint8x4: vec4<i32>,
  @location(4) sint16x2: vec2<i32>,
  @location(5) sint16x4: vec4<i32>,

  // 2-byte-aligned formats
  @location(6) uint8x2: vec2<u32>,
  @location(7) sint8x2: vec2<i32>,
}

@vertex
fn vertex_block_1(v_in: AttributeBlock1) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all uint into one checksum value.
  var all_uint: u32 = 0;
  all_uint = accumulate_uint(all_uint, v_in.uint8x2.x);
  all_uint = accumulate_uint(all_uint, v_in.uint8x2.y);

  all_uint = accumulate_uint(all_uint, v_in.uint8x4.x);
  all_uint = accumulate_uint(all_uint, v_in.uint8x4.y);
  all_uint = accumulate_uint(all_uint, v_in.uint8x4.z);
  all_uint = accumulate_uint(all_uint, v_in.uint8x4.w);

  all_uint = accumulate_uint(all_uint, v_in.uint16x2.x);
  all_uint = accumulate_uint(all_uint, v_in.uint16x2.y);

  all_uint = accumulate_uint(all_uint, v_in.uint16x4.x);
  all_uint = accumulate_uint(all_uint, v_in.uint16x4.y);
  all_uint = accumulate_uint(all_uint, v_in.uint16x4.z);
  all_uint = accumulate_uint(all_uint, v_in.uint16x4.w);

  checksums[index_uint] = f32(all_uint);

  // Accumulate all sint into one checksum value.
  var all_sint: i32 = 0;
  all_sint = accumulate_sint(all_sint, v_in.sint8x2.x);
  all_sint = accumulate_sint(all_sint, v_in.sint8x2.y);

  all_sint = accumulate_sint(all_sint, v_in.sint8x4.x);
  all_sint = accumulate_sint(all_sint, v_in.sint8x4.y);
  all_sint = accumulate_sint(all_sint, v_in.sint8x4.z);
  all_sint = accumulate_sint(all_sint, v_in.sint8x4.w);

  all_sint = accumulate_sint(all_sint, v_in.sint16x2.x);
  all_sint = accumulate_sint(all_sint, v_in.sint16x2.y);

  all_sint = accumulate_sint(all_sint, v_in.sint16x4.x);
  all_sint = accumulate_sint(all_sint, v_in.sint16x4.y);
  all_sint = accumulate_sint(all_sint, v_in.sint16x4.z);
  all_sint = accumulate_sint(all_sint, v_in.sint16x4.w);

  checksums[index_sint] = f32(all_sint);

  return vec4(0.0);
}

struct AttributeBlock2 {
  @location(0) uint32: u32,
  @location(1) uint32x2: vec2<u32>,
  @location(2) uint32x3: vec3<u32>,
  @location(3) uint32x4: vec4<u32>,
}

@vertex
fn vertex_block_2(v_in: AttributeBlock2) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all uint into one checksum value.
  var all_uint: u32 = 0;
  all_uint = accumulate_uint(all_uint, v_in.uint32);

  all_uint = accumulate_uint(all_uint, v_in.uint32x2.x);
  all_uint = accumulate_uint(all_uint, v_in.uint32x2.y);

  all_uint = accumulate_uint(all_uint, v_in.uint32x3.x);
  all_uint = accumulate_uint(all_uint, v_in.uint32x3.y);
  all_uint = accumulate_uint(all_uint, v_in.uint32x3.z);

  all_uint = accumulate_uint(all_uint, v_in.uint32x4.x);
  all_uint = accumulate_uint(all_uint, v_in.uint32x4.y);
  all_uint = accumulate_uint(all_uint, v_in.uint32x4.z);
  all_uint = accumulate_uint(all_uint, v_in.uint32x4.w);

  checksums[index_uint] = f32(all_uint);

  return vec4(0.0);
}

struct AttributeBlock3 {
  @location(0) sint32: i32,
  @location(1) sint32x2: vec2<i32>,
  @location(2) sint32x3: vec3<i32>,
  @location(3) sint32x4: vec4<i32>,
}

@vertex
fn vertex_block_3(v_in: AttributeBlock3) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all sint into one checksum value.
  var all_sint: i32 = 0;
  all_sint = accumulate_sint(all_sint, v_in.sint32);

  all_sint = accumulate_sint(all_sint, v_in.sint32x2.x);
  all_sint = accumulate_sint(all_sint, v_in.sint32x2.y);

  all_sint = accumulate_sint(all_sint, v_in.sint32x3.x);
  all_sint = accumulate_sint(all_sint, v_in.sint32x3.y);
  all_sint = accumulate_sint(all_sint, v_in.sint32x3.z);

  all_sint = accumulate_sint(all_sint, v_in.sint32x4.x);
  all_sint = accumulate_sint(all_sint, v_in.sint32x4.y);
  all_sint = accumulate_sint(all_sint, v_in.sint32x4.z);
  all_sint = accumulate_sint(all_sint, v_in.sint32x4.w);

  checksums[index_sint] = f32(all_sint);

  return vec4(0.0);
}

struct AttributeBlock4{
  @location(0) float32: f32,
  @location(1) float32x2: vec2<f32>,
  @location(2) float32x3: vec3<f32>,
  @location(3) float32x4: vec4<f32>,
  @location(4) float16x2: vec2<f32>,
  @location(5) float16x4: vec4<f32>,
  @location(6) float16: f32,
}

@vertex
fn vertex_block_4(v_in: AttributeBlock4) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all float32 into one checksum value.
  var all_float32: f32 = 0.0;
  all_float32 = accumulate_float32(all_float32, v_in.float32);

  all_float32 = accumulate_float32(all_float32, v_in.float32x2.x);
  all_float32 = accumulate_float32(all_float32, v_in.float32x2.y);

  all_float32 = accumulate_float32(all_float32, v_in.float32x3.x);
  all_float32 = accumulate_float32(all_float32, v_in.float32x3.y);
  all_float32 = accumulate_float32(all_float32, v_in.float32x3.z);

  all_float32 = accumulate_float32(all_float32, v_in.float32x4.x);
  all_float32 = accumulate_float32(all_float32, v_in.float32x4.y);
  all_float32 = accumulate_float32(all_float32, v_in.float32x4.z);
  all_float32 = accumulate_float32(all_float32, v_in.float32x4.w);

  checksums[index_float32] = f32(all_float32);

  // Accumulate all float16 into one checksum value.
  var all_float16: f32 = 0.0;
  all_float16 = accumulate_float16(all_float16, v_in.float16x2.x);
  all_float16 = accumulate_float16(all_float16, v_in.float16x2.y);

  all_float16 = accumulate_float16(all_float16, v_in.float16x4.x);
  all_float16 = accumulate_float16(all_float16, v_in.float16x4.y);
  all_float16 = accumulate_float16(all_float16, v_in.float16x4.z);
  all_float16 = accumulate_float16(all_float16, v_in.float16x4.w);

  all_float16 = accumulate_float16(all_float16, v_in.float16);

  checksums[index_float16] = f32(all_float16);

  return vec4(0.0);
}

struct AttributeBlock5{
  @location(0) unorm10_10_10_2: vec4<f32>,
}

@vertex
fn vertex_block_5(v_in: AttributeBlock5) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all unorm into one checksum value.
  var all_unorm: f32 = 0.0;
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm10_10_10_2.x);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm10_10_10_2.y);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm10_10_10_2.z);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm10_10_10_2.w);

  checksums[index_unorm] = f32(all_unorm);

  return vec4(0.0);
}

struct AttributeBlock6 {
  @location(0) uint16: u32,
  @location(1) sint16: i32,
  @location(2) unorm16: f32,
  @location(3) snorm16: f32,
  @location(4) uint8: u32,
  @location(5) sint8: i32,
  @location(6) unorm8: f32,
  @location(7) snorm8: f32,
}

@vertex
fn vertex_block_6(v_in: AttributeBlock6) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all unorm into one checksum value.
  var all_unorm: f32 = 0.0;
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm16);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8);
  checksums[index_unorm] = f32(all_unorm);

  // Accumulate all snorm into one checksum value.
  var all_snorm: f32 = 0.0;
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm16);
  all_snorm = accumulate_snorm(all_snorm, v_in.snorm8);
  checksums[index_snorm] = f32(all_snorm);

  // Accumulate all uint into one checksum value.
  var all_uint: u32 = 0;
  all_uint = accumulate_uint(all_uint, v_in.uint16);
  all_uint = accumulate_uint(all_uint, v_in.uint8);
  checksums[index_uint] = f32(all_uint);

  // Accumulate all sint into one checksum value.
  var all_sint: i32 = 0;
  all_sint = accumulate_sint(all_sint, v_in.sint16);
  all_sint = accumulate_sint(all_sint, v_in.sint8);
  checksums[index_sint] = f32(all_sint);

  return vec4(0.0);
}

struct AttributeBlock7 {
  @location(0) unorm8x4_bgra: vec4<f32>,
}

@vertex
fn vertex_block_7(v_in: AttributeBlock7) -> @builtin(position) vec4<f32>
{
  init_checksums();

  // Accumulate all unorm into one checksum value.
  var all_unorm: f32 = 0.0;
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4_bgra.r);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4_bgra.g);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4_bgra.b);
  all_unorm = accumulate_unorm(all_unorm, v_in.unorm8x4_bgra.a);

  checksums[index_unorm] = f32(all_unorm);

  return vec4(0.0);
}

fn accumulate_uint(accum: u32, val: u32) -> u32 {
  return accum + val;
}

fn accumulate_sint(accum: i32, val: i32) -> i32 {
  return accum + val;
}

fn accumulate_unorm(accum: f32, val: f32) -> f32 {
  return accum + val;
}

fn accumulate_snorm(accum: f32, val: f32) -> f32 {
  return accum + val;
}

fn accumulate_float16(accum: f32, val: f32) -> f32 {
  return accum + val;
}

fn accumulate_float32(accum: f32, val: f32) -> f32 {
  return accum + val;
}

@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}

var<private> private_variable: i64 = 1li;
const constant_variable: u64 = 20lu;

struct UniformCompatible {
   // Other types
   val_u32: u32,
   val_i32: i32,
   val_f32: f32,

   // u64
   val_u64: u64,
   val_u64_2: vec2<u64>,
   val_u64_3: vec3<u64>,
   val_u64_4: vec4<u64>,

   // i64
   val_i64: i64,
   val_i64_2: vec2<i64>,
   val_i64_3: vec3<i64>,
   val_i64_4: vec4<i64>,

   final_value: u64,
}

struct StorageCompatible {
   val_u64_array_2: array<u64, 2>,
   val_i64_array_2: array<i64, 2>,
}

@group(0) @binding(0)
var<uniform> input_uniform: UniformCompatible;
@group(0) @binding(1)
var<storage> input_storage: UniformCompatible;
@group(0) @binding(2)
var<storage> input_arrays: StorageCompatible;
@group(0) @binding(3)
var<storage, read_write> output: UniformCompatible;
@group(0) @binding(4)
var<storage, read_write> output_arrays: StorageCompatible;

fn int64_function(x: i64) -> i64 {
   var val: i64 = i64(constant_variable);
   // A number too big for i32
   val += 31li - 1002003004005006li + -0x7fffffffffffffffli;
   // Constructing an i64 from an AbstractInt
   val += val + i64(5);
   // Constructing a i64 from other types and other types from u64.
   val += i64(input_uniform.val_u32 + u32(val));
   val += i64(input_uniform.val_i32 + i32(val));
   val += i64(input_uniform.val_f32 + f32(val));
   // Constructing a vec3<i64> from a i64
   val += vec3<i64>(input_uniform.val_i64).z;
   // Bitcasting from u64 to i64
   val += bitcast<i64>(input_uniform.val_u64);
   val += bitcast<vec2<i64>>(input_uniform.val_u64_2).y;
   val += bitcast<vec3<i64>>(input_uniform.val_u64_3).z;
   val += bitcast<vec4<i64>>(input_uniform.val_u64_4).w;

   // Reading/writing to a uniform/storage buffer
   output.val_i64 = input_uniform.val_i64 + input_storage.val_i64;
   output.val_i64_2 = input_uniform.val_i64_2 + input_storage.val_i64_2;
   output.val_i64_3 = input_uniform.val_i64_3 + input_storage.val_i64_3;
   output.val_i64_4 = input_uniform.val_i64_4 + input_storage.val_i64_4;

   output_arrays.val_i64_array_2 = input_arrays.val_i64_array_2;

   // We make sure not to use 32 in these arguments, so it's clear in the results which are builtin
   // constants based on the size of the type, and which are arguments.

   // Numeric functions
   val += abs(val);
   val += clamp(val, val, val);
   //val += countLeadingZeros(val);
   //val += countOneBits(val);
   //val += countTrailingZeros(val);
   val += dot(vec2(val), vec2(val));
   //val += extractBits(val, 15u, 28u);
   //val += firstLeadingBit(val);
   //val += firstTrailingBit(val);
   //val += insertBits(val, 12li, 15u, 28u);
   val += max(val, val);
   val += min(val, val);
   //val += reverseBits(val);
   val += sign(val); // only for i64

   // Make sure all the variables are used.
   return val;
}

fn uint64_function(x: u64) -> u64 {
   var val: u64 = u64(constant_variable);
   // Numbers too big for u32 (u64::MAX)
   val += 31lu + 18446744073709551615lu - 0xfffffffffffffffflu;
   // Constructing a u64 from an AbstractInt
   val += val + u64(5);
   // Constructing a u64 from other types and other types from u64.
   val += u64(input_uniform.val_u32 + u32(val));
   val += u64(input_uniform.val_i32 + i32(val));
   val += u64(input_uniform.val_f32 + f32(val));
   // Constructing a vec3<u64> from a u64
   val += vec3<u64>(input_uniform.val_u64).z;
   // Bitcasting from i64 to u64
   val += bitcast<u64>(input_uniform.val_i64);
   val += bitcast<vec2<u64>>(input_uniform.val_i64_2).y;
   val += bitcast<vec3<u64>>(input_uniform.val_i64_3).z;
   val += bitcast<vec4<u64>>(input_uniform.val_i64_4).w;

   // Reading/writing to a uniform/storage buffer
   output.val_u64 = input_uniform.val_u64 + input_storage.val_u64;
   output.val_u64_2 = input_uniform.val_u64_2 + input_storage.val_u64_2;
   output.val_u64_3 = input_uniform.val_u64_3 + input_storage.val_u64_3;
   output.val_u64_4 = input_uniform.val_u64_4 + input_storage.val_u64_4;

   output_arrays.val_u64_array_2 = input_arrays.val_u64_array_2;

   // We make sure not to use 32 in these arguments, so it's clear in the results which are builtin
   // constants based on the size of the type, and which are arguments.

   // Numeric functions
   val += abs(val);
   val += clamp(val, val, val);
   //val += countLeadingZeros(val);
   //val += countOneBits(val);
   //val += countTrailingZeros(val);
   val += dot(vec2(val), vec2(val));
   //val += extractBits(val, 15u, 28u);
   //val += firstLeadingBit(val);
   //val += firstTrailingBit(val);
   //val += insertBits(val, 12lu, 15u, 28u);
   val += max(val, val);
   val += min(val, val);
   //val += reverseBits(val);

   // Make sure all the variables are used.
   return val;
}

@compute @workgroup_size(1)
fn main() {
   output.final_value = uint64_function(67lu) + bitcast<u64>(int64_function(60li));
}

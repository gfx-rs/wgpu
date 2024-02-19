var<private> private_variable: i64 = 1li;
const constant_variable: u64 = 20lu;

struct UniformCompatible {
   // Other types
   val_u32: u32,
   val_i32: i32,
   val_f32: f32,
   val_f64: f64,

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
   // A number too big for i32
   let val_1: i64 = 31li - 1002003004005006li;
   // Constructing an i64 from an AbstractInt
   let val_2 = val_1 + i64(5);
   // Constructing a i64 from other types.
   let val_3 = 
        i64(input_uniform.val_u32)
      + i64(input_uniform.val_i32)
      + i64(input_uniform.val_f32)
      + i64(input_uniform.val_f64);
   // Constructing a vec3<i64> from a i64
   let val_4 = vec3<i64>(input_uniform.val_i64);
   // Reading/writing to a uniform/storage buffer
   output.val_i64 = input_uniform.val_i64 + input_storage.val_i64;
   output.val_i64_2 = input_uniform.val_i64_2 + input_storage.val_i64_2;
   output.val_i64_3 = input_uniform.val_i64_3 + input_storage.val_i64_3;
   output.val_i64_4 = input_uniform.val_i64_4 + input_storage.val_i64_4;

   output_arrays.val_i64_array_2 = input_arrays.val_i64_array_2;
   // Make sure all the variables are used.
   return val_1 + val_2 + val_3 + i64(constant_variable) + 50li;
}

fn uint64_function(x: u64) -> u64 {
   // A number too big for u32
   let val_1: u64 = 31lu + 1002003004005006lu;
   // Constructing a u64 from an AbstractInt
   let val_2 = val_1 + u64(5);
   // Constructing a u64 from other types.
   let val_3 = 
        u64(input_uniform.val_u32)
      + u64(input_uniform.val_i32)
      + u64(input_uniform.val_f32)
      + u64(input_uniform.val_f64);
   // Constructing a vec3<u64> from a u64
   let val_4 = vec3<u64>(input_uniform.val_u64);
   // Reading/writing to a uniform/storage buffer
   output.val_u64 = input_uniform.val_u64 + input_storage.val_u64;
   output.val_u64_2 = input_uniform.val_u64_2 + input_storage.val_u64_2;
   output.val_u64_3 = input_uniform.val_u64_3 + input_storage.val_u64_3;
   output.val_u64_4 = input_uniform.val_u64_4 + input_storage.val_u64_4;

   output_arrays.val_u64_array_2 = input_arrays.val_u64_array_2;
   // Make sure all the variables are used.
   return val_1 + val_2 + val_3 + u64(constant_variable) + 50lu;
}

@compute @workgroup_size(1)
fn main() {
   uint64_function(67lu);
   int64_function(60li);
}

enable f16;
enable f16; //redundant directives are OK

var<private> private_variable: f16 = 1h;
const constant_variable: f16 = f16(15.2);

struct UniformCompatible {
   // Other types
   val_u32: u32,
   val_i32: i32,
   val_f32: f32,

   // f16
   val_f16: f16,
   val_f16_2: vec2<f16>,
   val_f16_3: vec3<f16>,
   val_f16_4: vec4<f16>,
   final_value: f16,
}

struct StorageCompatible {
   val_f16_array_2: array<f16, 2>,
   val_f16_array_2: array<f16, 2>,
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

fn f16_function(x: f16) -> f16 {
   var val: f16 = f16(constant_variable);
   // A number too big for f16
   val += 1h - 33333h;
   // Constructing an f16 from an AbstractInt
   val += val + f16(5.);
   // Constructing a f16 from other types and other types from f16.
   val += f16(input_uniform.val_f32 + f32(val));
   // Constructing a vec3<i64> from a i64
   val += vec3<f16>(input_uniform.val_f16).z;

   // Reading/writing to a uniform/storage buffer
   output.val_f16 = input_uniform.val_f16 + input_storage.val_f16;
   output.val_f16_2 = input_uniform.val_f16_2 + input_storage.val_f16_2;
   output.val_f16_3 = input_uniform.val_f16_3 + input_storage.val_f16_3;
   output.val_f16_4 = input_uniform.val_f16_4 + input_storage.val_f16_4;

   output_arrays.val_f16_array_2 = input_arrays.val_f16_array_2;

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
   return f16(1.0);
}

@compute @workgroup_size(1)
fn main() {
   output.final_value = f16_function(2h);
}


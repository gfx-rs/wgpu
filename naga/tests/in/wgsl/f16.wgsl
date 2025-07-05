enable f16;

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

   val_mat2x2: mat2x2<f16>,
   val_mat2x3: mat2x3<f16>,
   val_mat2x4: mat2x4<f16>,
   val_mat3x2: mat3x2<f16>,
   val_mat3x3: mat3x3<f16>,
   val_mat3x4: mat3x4<f16>,
   val_mat4x2: mat4x2<f16>,
   val_mat4x3: mat4x3<f16>,
   val_mat4x4: mat4x4<f16>,
}

struct StorageCompatible {
   val_f16_array_2: array<f16, 2>,
}

struct LayoutTest {
   scalar1: f16, scalar2: f16, v3: vec3<f16>, tuck_in: f16, scalar4: f16, larger: u32
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
   _ = private_variable;
   var l: LayoutTest;

   var val: f16 = f16(constant_variable);
   // A number too big for f16
   val += 1h - 33333h;
   // Constructing an f16 from an AbstractInt
   val += val + f16(5.);
   // Constructing a f16 from other types and other types from f16.
   val += f16(input_uniform.val_f32 + f32(val));
   // Constructing a vec3<i64> from a i64
   val += vec3<f16>(input_uniform.val_f16).z;

   // Cast min and max finite f16 literals to other types. Max value should convert
   // exactly to other types, but min (or any negative) should clamp to zero for u32.
   output.val_i32 = i32(65504h);
   output.val_i32 = i32(-65504h);
   output.val_u32 = u32(65504h);
   output.val_u32 = u32(-65504h);
   output.val_f32 = f32(65504h);
   output.val_f32 = f32(-65504h);

   // Reading/writing to a uniform/storage buffer
   output.val_f16 = input_uniform.val_f16 + input_storage.val_f16;
   output.val_f16_2 = input_uniform.val_f16_2 + input_storage.val_f16_2;
   output.val_f16_3 = input_uniform.val_f16_3 + input_storage.val_f16_3;
   output.val_f16_4 = input_uniform.val_f16_4 + input_storage.val_f16_4;

   output.val_mat2x2 = input_uniform.val_mat2x2 + input_storage.val_mat2x2;
   output.val_mat2x3 = input_uniform.val_mat2x3 + input_storage.val_mat2x3;
   output.val_mat2x4 = input_uniform.val_mat2x4 + input_storage.val_mat2x4;
   output.val_mat3x2 = input_uniform.val_mat3x2 + input_storage.val_mat3x2;
   output.val_mat3x3 = input_uniform.val_mat3x3 + input_storage.val_mat3x3;
   output.val_mat3x4 = input_uniform.val_mat3x4 + input_storage.val_mat3x4;
   output.val_mat4x2 = input_uniform.val_mat4x2 + input_storage.val_mat4x2;
   output.val_mat4x3 = input_uniform.val_mat4x3 + input_storage.val_mat4x3;
   output.val_mat4x4 = input_uniform.val_mat4x4 + input_storage.val_mat4x4;

   output_arrays.val_f16_array_2 = input_arrays.val_f16_array_2;

   // We make sure not to use 32 in these arguments, so it's clear in the results which are builtin
   // constants based on the size of the type, and which are arguments.

   // Numeric functions
   val += abs(val);
   val += clamp(val, val, val);
   val += dot(vec2(val), vec2(val));
   val += max(val, val);
   val += min(val, val);
   val += sign(val);
   
   val += f16(1.0);

   // We use the shorthand aliases here to ensure the aliases
   // work correctly.

   // Cast vectors to/from f32
   let float_vec2 = vec2f(input_uniform.val_f16_2);
   output.val_f16_2 = vec2h(float_vec2);

   let float_vec3 = vec3f(input_uniform.val_f16_3);
   output.val_f16_3 = vec3h(float_vec3);

   let float_vec4 = vec4f(input_uniform.val_f16_4);
   output.val_f16_4 = vec4h(float_vec4);

   // Cast matrices to/from f32
   output.val_mat2x2 = mat2x2h(mat2x2f(input_uniform.val_mat2x2));
   output.val_mat2x3 = mat2x3h(mat2x3f(input_uniform.val_mat2x3));
   output.val_mat2x4 = mat2x4h(mat2x4f(input_uniform.val_mat2x4));
   output.val_mat3x2 = mat3x2h(mat3x2f(input_uniform.val_mat3x2));
   output.val_mat3x3 = mat3x3h(mat3x3f(input_uniform.val_mat3x3));
   output.val_mat3x4 = mat3x4h(mat3x4f(input_uniform.val_mat3x4));
   output.val_mat4x2 = mat4x2h(mat4x2f(input_uniform.val_mat4x2));
   output.val_mat4x3 = mat4x3h(mat4x3f(input_uniform.val_mat4x3));
   output.val_mat4x4 = mat4x4h(mat4x4f(input_uniform.val_mat4x4));

   // Make sure all the variables are used.
   return val;
}

@compute @workgroup_size(1)
fn main() {
   output.final_value = f16_function(2h);
}

enable wgpu_int16;
enable f16;

var<private> private_variable: i16 = i16(1);
const constant_variable: u16 = u16(20);

// f16 can represent up to 65504, but i16 max is 32767; the cast must clamp.
const f16_to_i16_clamped: i16 = i16(f16(33000.0));

struct UniformCompatible {
   // Other types
   val_u32: u32,
   val_i32: i32,
   val_f32: f32,

   // u16
   val_u16: u16,
   val_u16_2: vec2<u16>,
   val_u16_3: vec3<u16>,
   val_u16_4: vec4<u16>,

   // i16
   val_i16: i16,
   val_i16_2: vec2<i16>,
   val_i16_3: vec3<i16>,
   val_i16_4: vec4<i16>,

   final_value: u16,
}

struct StorageCompatible {
   val_u16_array_2: array<u16, 2>,
   val_i16_array_2: array<i16, 2>,
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

// Test workgroup variable
var<workgroup> shared_val: u16;

fn int16_function(x: i16) -> i16 {
   _ = private_variable;
   var val: i16 = i16(constant_variable);

   // Constructing an i16 from an AbstractInt
   val = val + i16(5);
   // Constructing an i16 from other types
   val = val + i16(input_uniform.val_u32);
   val = val + i16(input_uniform.val_i32);
   // Constructing a vec3<i16> from an i16
   val = val + vec3<i16>(input_uniform.val_i16).z;

   // Reading/writing to a uniform/storage buffer
   output.val_i16 = input_uniform.val_i16 + input_storage.val_i16;
   output.val_i16_2 = input_uniform.val_i16_2 + input_storage.val_i16_2;
   output.val_i16_3 = input_uniform.val_i16_3 + input_storage.val_i16_3;
   output.val_i16_4 = input_uniform.val_i16_4 + input_storage.val_i16_4;

   output_arrays.val_i16_array_2 = input_arrays.val_i16_array_2;

   // Numeric builtin functions
   val = abs(val);
   val = max(val, val);
   val = min(val, val);
   val = clamp(val, val, val);
   val = sign(val);

   // Binary arithmetic operators
   val = val - i16(1);
   val = val * i16(2);
   val = val / i16(3);
   val = val % i16(4);

   // Bitwise operators
   val = val & i16(0xFF);
   val = val | i16(0x10);
   val = val ^ i16(0x01);

   // Shift operators
   val = val << 2u;
   val = val >> 1u;

   // Unary negation
   val = -val;

   // Comparison operators
   let cmp_lt = val < i16(0);
   let cmp_le = val <= i16(0);
   let cmp_gt = val > i16(0);
   let cmp_ge = val >= i16(0);
   let cmp_eq = val == i16(0);
   let cmp_ne = val != i16(0);

   // Select
   val = select(i16(1), i16(2), cmp_lt);

   // Local array variable, construction, and indexing
   var arr: array<i16, 4> = array<i16, 4>(i16(1), i16(2), i16(3), i16(4));
   arr[0] = val;
   val = arr[1];
   // Indexing via u16
   let u16_idx = u16(1);
   val = arr[u16_idx];

   // Cast to/from other types
   output.val_u32 = u32(val);
   output.val_i32 = i32(val);
   output.val_f32 = f32(val);
   val = i16(output.val_u32);

   // Bitcast between i16 and u16
   let as_unsigned = bitcast<u16>(val);
   val = bitcast<i16>(as_unsigned);

   // Vector arithmetic
   let v = input_uniform.val_i16_2 + input_uniform.val_i16_2;
   let v2 = v * vec2<i16>(i16(2));
   output.val_i16_2 = v2;

   return val;
}

fn uint16_function(x: u16) -> u16 {
   var val: u16 = u16(constant_variable);

   // Constructing a u16 from an AbstractInt
   val = val + u16(5);
   // Constructing a u16 from other types
   val = val + u16(input_uniform.val_u32);
   val = val + u16(input_uniform.val_i32);
   // Constructing a vec3<u16> from a u16
   val = val + vec3<u16>(input_uniform.val_u16).z;

   // Reading/writing to a uniform/storage buffer
   output.val_u16 = input_uniform.val_u16 + input_storage.val_u16;
   output.val_u16_2 = input_uniform.val_u16_2 + input_storage.val_u16_2;
   output.val_u16_3 = input_uniform.val_u16_3 + input_storage.val_u16_3;
   output.val_u16_4 = input_uniform.val_u16_4 + input_storage.val_u16_4;

   output_arrays.val_u16_array_2 = input_arrays.val_u16_array_2;

   // Numeric builtin functions
   val = abs(val);
   val = max(val, val);
   val = min(val, val);
   val = clamp(val, val, val);

   // Binary arithmetic operators
   val = val - u16(1);
   val = val * u16(2);
   val = val / u16(3);
   val = val % u16(4);

   // Bitwise operators
   val = val & u16(0xFF);
   val = val | u16(0x10);
   val = val ^ u16(0x01);

   // Cast to/from other types
   output.val_u32 = u32(val);
   output.val_i32 = i32(val);
   output.val_f32 = f32(val);
   val = u16(output.val_u32);

   return val;
}

@compute @workgroup_size(64)
fn main(
   @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
) {
   shared_val = u16(0);
   output.final_value = uint16_function(u16(67)) + u16(int16_function(i16(60)));

   // Subgroup operations on i16
   var sg_val: i16 = i16(subgroup_invocation_id);
   sg_val = subgroupAdd(sg_val);
   sg_val = subgroupMul(sg_val);
   sg_val = subgroupMin(sg_val);
   sg_val = subgroupMax(sg_val);
   // Note: subgroupAnd/Or/Xor are omitted because HLSL's
   // WaveActiveBitAnd/Or/Xor don't support 16-bit types.
   sg_val = subgroupExclusiveAdd(sg_val);
   sg_val = subgroupInclusiveAdd(sg_val);
   sg_val = subgroupBroadcastFirst(sg_val);
   sg_val = subgroupBroadcast(sg_val, 4u);

   // Subgroup operations on u16
   var sg_uval: u16 = u16(subgroup_invocation_id);
   sg_uval = subgroupAdd(sg_uval);
   sg_uval = subgroupMin(sg_uval);
   sg_uval = subgroupMax(sg_uval);

   output.val_i16 = sg_val;
   output.val_u16 = sg_uval;
}

enable wgpu_int16;

struct UniformCompatible {
    val_u32_: u32,
    val_i32_: i32,
    val_f32_: f32,
    val_u16_: u16,
    val_u16_2_: vec2<u16>,
    val_u16_3_: vec3<u16>,
    val_u16_4_: vec4<u16>,
    val_i16_: i16,
    val_i16_2_: vec2<i16>,
    val_i16_3_: vec3<i16>,
    val_i16_4_: vec4<i16>,
    final_value: u16,
}

struct StorageCompatible {
    val_u16_array_2_: array<u16, 2>,
    val_i16_array_2_: array<i16, 2>,
}

const constant_variable: u16 = u16(20);
const f16_to_i16_clamped: i16 = i16(32767);

var<private> private_variable: i16 = i16(1);
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
var<workgroup> shared_val: u16;

fn int16_function(x: i16) -> i16 {
    var val: i16 = i16(20);
    var arr: array<i16, 4> = array<i16, 4>(i16(1), i16(2), i16(3), i16(4));

    let phony = private_variable;
    let _e5 = val;
    val = (_e5 + i16(5));
    let _e8 = val;
    let _e11 = input_uniform.val_u32_;
    val = (_e8 + i16(_e11));
    let _e14 = val;
    let _e17 = input_uniform.val_i32_;
    val = (_e14 + i16(_e17));
    let _e20 = val;
    let _e23 = input_uniform.val_i16_;
    val = (_e20 + vec3(_e23).z);
    let _e31 = input_uniform.val_i16_;
    let _e34 = input_storage.val_i16_;
    output.val_i16_ = (_e31 + _e34);
    let _e40 = input_uniform.val_i16_2_;
    let _e43 = input_storage.val_i16_2_;
    output.val_i16_2_ = (_e40 + _e43);
    let _e49 = input_uniform.val_i16_3_;
    let _e52 = input_storage.val_i16_3_;
    output.val_i16_3_ = (_e49 + _e52);
    let _e58 = input_uniform.val_i16_4_;
    let _e61 = input_storage.val_i16_4_;
    output.val_i16_4_ = (_e58 + _e61);
    let _e67 = input_arrays.val_i16_array_2_;
    output_arrays.val_i16_array_2_ = _e67;
    let _e68 = val;
    val = abs(_e68);
    let _e70 = val;
    let _e71 = val;
    val = max(_e70, _e71);
    let _e73 = val;
    let _e74 = val;
    val = min(_e73, _e74);
    let _e76 = val;
    let _e77 = val;
    let _e78 = val;
    val = clamp(_e76, _e77, _e78);
    let _e80 = val;
    val = sign(_e80);
    let _e82 = val;
    val = (_e82 - i16(1));
    let _e85 = val;
    val = (_e85 * i16(2));
    let _e88 = val;
    val = (_e88 / i16(3));
    let _e91 = val;
    val = (_e91 % i16(4));
    let _e94 = val;
    val = (_e94 & i16(255));
    let _e97 = val;
    val = (_e97 | i16(16));
    let _e100 = val;
    val = (_e100 ^ i16(1));
    let _e103 = val;
    val = (_e103 << 2u);
    let _e106 = val;
    val = (_e106 >> 1u);
    let _e109 = val;
    val = -(_e109);
    let _e111 = val;
    let cmp_lt = (_e111 < i16(0));
    let _e114 = val;
    let cmp_le = (_e114 <= i16(0));
    let _e117 = val;
    let cmp_gt = (_e117 > i16(0));
    let _e120 = val;
    let cmp_ge = (_e120 >= i16(0));
    let _e123 = val;
    let cmp_eq = (_e123 == i16(0));
    let _e126 = val;
    let cmp_ne = (_e126 != i16(0));
    val = select(i16(1), i16(2), cmp_lt);
    let _e139 = val;
    arr[0] = _e139;
    let _e141 = arr[1];
    val = _e141;
    let _e144 = arr[u16(1)];
    val = _e144;
    let _e147 = val;
    output.val_u32_ = u32(_e147);
    let _e151 = val;
    output.val_i32_ = i32(_e151);
    let _e155 = val;
    output.val_f32_ = f32(_e155);
    let _e159 = output.val_u32_;
    val = i16(_e159);
    let _e161 = val;
    let as_unsigned = bitcast<u16>(_e161);
    val = bitcast<i16>(as_unsigned);
    let _e166 = input_uniform.val_i16_2_;
    let _e169 = input_uniform.val_i16_2_;
    let v = (_e166 + _e169);
    let v2_ = (v * vec2(i16(2)));
    output.val_i16_2_ = v2_;
    let _e176 = val;
    return _e176;
}

fn uint16_function(x_1: u16) -> u16 {
    var val_1: u16 = u16(20);

    let _e3 = val_1;
    val_1 = (_e3 + u16(5));
    let _e6 = val_1;
    let _e9 = input_uniform.val_u32_;
    val_1 = (_e6 + u16(_e9));
    let _e12 = val_1;
    let _e15 = input_uniform.val_i32_;
    val_1 = (_e12 + u16(_e15));
    let _e18 = val_1;
    let _e21 = input_uniform.val_u16_;
    val_1 = (_e18 + vec3(_e21).z);
    let _e29 = input_uniform.val_u16_;
    let _e32 = input_storage.val_u16_;
    output.val_u16_ = (_e29 + _e32);
    let _e38 = input_uniform.val_u16_2_;
    let _e41 = input_storage.val_u16_2_;
    output.val_u16_2_ = (_e38 + _e41);
    let _e47 = input_uniform.val_u16_3_;
    let _e50 = input_storage.val_u16_3_;
    output.val_u16_3_ = (_e47 + _e50);
    let _e56 = input_uniform.val_u16_4_;
    let _e59 = input_storage.val_u16_4_;
    output.val_u16_4_ = (_e56 + _e59);
    let _e65 = input_arrays.val_u16_array_2_;
    output_arrays.val_u16_array_2_ = _e65;
    let _e66 = val_1;
    val_1 = abs(_e66);
    let _e68 = val_1;
    let _e69 = val_1;
    val_1 = max(_e68, _e69);
    let _e71 = val_1;
    let _e72 = val_1;
    val_1 = min(_e71, _e72);
    let _e74 = val_1;
    let _e75 = val_1;
    let _e76 = val_1;
    val_1 = clamp(_e74, _e75, _e76);
    let _e78 = val_1;
    val_1 = (_e78 - u16(1));
    let _e81 = val_1;
    val_1 = (_e81 * u16(2));
    let _e84 = val_1;
    val_1 = (_e84 / u16(3));
    let _e87 = val_1;
    val_1 = (_e87 % u16(4));
    let _e90 = val_1;
    val_1 = (_e90 & u16(255));
    let _e93 = val_1;
    val_1 = (_e93 | u16(16));
    let _e96 = val_1;
    val_1 = (_e96 ^ u16(1));
    let _e101 = val_1;
    output.val_u32_ = u32(_e101);
    let _e105 = val_1;
    output.val_i32_ = i32(_e105);
    let _e109 = val_1;
    output.val_f32_ = f32(_e109);
    let _e113 = output.val_u32_;
    val_1 = u16(_e113);
    let _e115 = val_1;
    return _e115;
}

@compute @workgroup_size(64, 1, 1) 
fn main(@builtin(subgroup_invocation_id) subgroup_invocation_id: u32) {
    var sg_val: i16;
    var sg_uval: u16;

    shared_val = u16(0);
    let _e6 = uint16_function(u16(67));
    let _e8 = int16_function(i16(60));
    output.final_value = (_e6 + u16(_e8));
    sg_val = i16(subgroup_invocation_id);
    let _e13 = sg_val;
    let _e14 = subgroupAdd(_e13);
    sg_val = _e14;
    let _e15 = sg_val;
    let _e16 = subgroupMul(_e15);
    sg_val = _e16;
    let _e17 = sg_val;
    let _e18 = subgroupMin(_e17);
    sg_val = _e18;
    let _e19 = sg_val;
    let _e20 = subgroupMax(_e19);
    sg_val = _e20;
    let _e21 = sg_val;
    let _e22 = subgroupExclusiveAdd(_e21);
    sg_val = _e22;
    let _e23 = sg_val;
    let _e24 = subgroupInclusiveAdd(_e23);
    sg_val = _e24;
    let _e25 = sg_val;
    let _e26 = subgroupBroadcastFirst(_e25);
    sg_val = _e26;
    let _e27 = sg_val;
    let _e29 = subgroupBroadcast(_e27, 4u);
    sg_val = _e29;
    sg_uval = u16(subgroup_invocation_id);
    let _e32 = sg_uval;
    let _e33 = subgroupAdd(_e32);
    sg_uval = _e33;
    let _e34 = sg_uval;
    let _e35 = subgroupMin(_e34);
    sg_uval = _e35;
    let _e36 = sg_uval;
    let _e37 = subgroupMax(_e36);
    sg_uval = _e37;
    let _e40 = sg_val;
    output.val_i16_ = _e40;
    let _e43 = sg_uval;
    output.val_u16_ = _e43;
    return;
}

enable f16;

struct UniformCompatible {
    val_u32_: u32,
    val_i32_: i32,
    val_f32_: f32,
    val_f16_: f16,
    val_f16_2_: vec2<f16>,
    val_f16_3_: vec3<f16>,
    val_f16_4_: vec4<f16>,
    final_value: f16,
    val_mat2x2_: mat2x2<f16>,
    val_mat2x3_: mat2x3<f16>,
    val_mat2x4_: mat2x4<f16>,
    val_mat3x2_: mat3x2<f16>,
    val_mat3x3_: mat3x3<f16>,
    val_mat3x4_: mat3x4<f16>,
    val_mat4x2_: mat4x2<f16>,
    val_mat4x3_: mat4x3<f16>,
    val_mat4x4_: mat4x4<f16>,
}

struct StorageCompatible {
    val_f16_array_2_: array<f16, 2>,
}

struct LayoutTest {
    scalar1_: f16,
    scalar2_: f16,
    v3_: vec3<f16>,
    tuck_in: f16,
    scalar4_: f16,
    larger: u32,
}

const constant_variable: f16 = 15.203125h;

var<private> private_variable: f16 = 1h;
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
    var l: LayoutTest;
    var val: f16 = 15.203125h;

    let phony = private_variable;
    let _e6 = val;
    val = (_e6 + -33344h);
    let _e8 = val;
    let _e11 = val;
    val = (_e11 + (_e8 + 5h));
    let _e15 = input_uniform.val_f32_;
    let _e16 = val;
    let _e20 = val;
    val = (_e20 + f16((_e15 + f32(_e16))));
    let _e24 = input_uniform.val_f16_;
    let _e27 = val;
    val = (_e27 + vec3(_e24).z);
    output.val_i32_ = 65504i;
    output.val_i32_ = -65504i;
    output.val_u32_ = 65504u;
    output.val_u32_ = 0u;
    output.val_f32_ = 65504f;
    output.val_f32_ = -65504f;
    let _e51 = input_uniform.val_f16_;
    let _e54 = input_storage.val_f16_;
    output.val_f16_ = (_e51 + _e54);
    let _e60 = input_uniform.val_f16_2_;
    let _e63 = input_storage.val_f16_2_;
    output.val_f16_2_ = (_e60 + _e63);
    let _e69 = input_uniform.val_f16_3_;
    let _e72 = input_storage.val_f16_3_;
    output.val_f16_3_ = (_e69 + _e72);
    let _e78 = input_uniform.val_f16_4_;
    let _e81 = input_storage.val_f16_4_;
    output.val_f16_4_ = (_e78 + _e81);
    let _e87 = input_uniform.val_mat2x2_;
    let _e90 = input_storage.val_mat2x2_;
    output.val_mat2x2_ = (_e87 + _e90);
    let _e96 = input_uniform.val_mat2x3_;
    let _e99 = input_storage.val_mat2x3_;
    output.val_mat2x3_ = (_e96 + _e99);
    let _e105 = input_uniform.val_mat2x4_;
    let _e108 = input_storage.val_mat2x4_;
    output.val_mat2x4_ = (_e105 + _e108);
    let _e114 = input_uniform.val_mat3x2_;
    let _e117 = input_storage.val_mat3x2_;
    output.val_mat3x2_ = (_e114 + _e117);
    let _e123 = input_uniform.val_mat3x3_;
    let _e126 = input_storage.val_mat3x3_;
    output.val_mat3x3_ = (_e123 + _e126);
    let _e132 = input_uniform.val_mat3x4_;
    let _e135 = input_storage.val_mat3x4_;
    output.val_mat3x4_ = (_e132 + _e135);
    let _e141 = input_uniform.val_mat4x2_;
    let _e144 = input_storage.val_mat4x2_;
    output.val_mat4x2_ = (_e141 + _e144);
    let _e150 = input_uniform.val_mat4x3_;
    let _e153 = input_storage.val_mat4x3_;
    output.val_mat4x3_ = (_e150 + _e153);
    let _e159 = input_uniform.val_mat4x4_;
    let _e162 = input_storage.val_mat4x4_;
    output.val_mat4x4_ = (_e159 + _e162);
    let _e168 = input_arrays.val_f16_array_2_;
    output_arrays.val_f16_array_2_ = _e168;
    let _e169 = val;
    let _e171 = val;
    val = (_e171 + abs(_e169));
    let _e173 = val;
    let _e174 = val;
    let _e175 = val;
    let _e177 = val;
    val = (_e177 + clamp(_e173, _e174, _e175));
    let _e179 = val;
    let _e181 = val;
    let _e184 = val;
    val = (_e184 + dot(vec2(_e179), vec2(_e181)));
    let _e186 = val;
    let _e187 = val;
    let _e189 = val;
    val = (_e189 + max(_e186, _e187));
    let _e191 = val;
    let _e192 = val;
    let _e194 = val;
    val = (_e194 + min(_e191, _e192));
    let _e196 = val;
    let _e198 = val;
    val = (_e198 + sign(_e196));
    let _e201 = val;
    val = (_e201 + 1h);
    let _e205 = input_uniform.val_f16_2_;
    let float_vec2_ = vec2<f32>(_e205);
    output.val_f16_2_ = vec2<f16>(float_vec2_);
    let _e212 = input_uniform.val_f16_3_;
    let float_vec3_ = vec3<f32>(_e212);
    output.val_f16_3_ = vec3<f16>(float_vec3_);
    let _e219 = input_uniform.val_f16_4_;
    let float_vec4_ = vec4<f32>(_e219);
    output.val_f16_4_ = vec4<f16>(float_vec4_);
    let _e228 = input_uniform.val_mat2x2_;
    output.val_mat2x2_ = mat2x2<f16>(mat2x2<f32>(_e228));
    let _e235 = input_uniform.val_mat2x3_;
    output.val_mat2x3_ = mat2x3<f16>(mat2x3<f32>(_e235));
    let _e242 = input_uniform.val_mat2x4_;
    output.val_mat2x4_ = mat2x4<f16>(mat2x4<f32>(_e242));
    let _e249 = input_uniform.val_mat3x2_;
    output.val_mat3x2_ = mat3x2<f16>(mat3x2<f32>(_e249));
    let _e256 = input_uniform.val_mat3x3_;
    output.val_mat3x3_ = mat3x3<f16>(mat3x3<f32>(_e256));
    let _e263 = input_uniform.val_mat3x4_;
    output.val_mat3x4_ = mat3x4<f16>(mat3x4<f32>(_e263));
    let _e270 = input_uniform.val_mat4x2_;
    output.val_mat4x2_ = mat4x2<f16>(mat4x2<f32>(_e270));
    let _e277 = input_uniform.val_mat4x3_;
    output.val_mat4x3_ = mat4x3<f16>(mat4x3<f32>(_e277));
    let _e284 = input_uniform.val_mat4x4_;
    output.val_mat4x4_ = mat4x4<f16>(mat4x4<f32>(_e284));
    let _e287 = val;
    return _e287;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e3 = f16_function(2h);
    output.final_value = _e3;
    return;
}

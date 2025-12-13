struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct UniformCompatible {
    uint val_u32_;
    int val_i32_;
    float val_f32_;
    half val_f16_;
    half2 val_f16_2_;
    int _pad5_0;
    half3 val_f16_3_;
    half4 val_f16_4_;
    half final_value;
    half2 val_mat2x2__0; half2 val_mat2x2__1;
    int _pad9_0;
    row_major half2x3 val_mat2x3_;
    row_major half2x4 val_mat2x4_;
    half2 val_mat3x2__0; half2 val_mat3x2__1; half2 val_mat3x2__2;
    int _pad12_0;
    row_major half3x3 val_mat3x3_;
    row_major half3x4 val_mat3x4_;
    half2 val_mat4x2__0; half2 val_mat4x2__1; half2 val_mat4x2__2; half2 val_mat4x2__3;
    row_major half4x3 val_mat4x3_;
    row_major half4x4 val_mat4x4_;
};

struct StorageCompatible {
    half val_f16_array_2_[2];
};

struct LayoutTest {
    half scalar1_;
    half scalar2_;
    int _pad2_0;
    half3 v3_;
    half tuck_in;
    half scalar4_;
    uint larger;
};

static const half constant_variable = 15.203125h;

static half private_variable = 1.0h;
cbuffer input_uniform : register(b0) { UniformCompatible input_uniform; }
ByteAddressBuffer input_storage : register(t1);
ByteAddressBuffer input_arrays : register(t2);
RWByteAddressBuffer output : register(u3);
RWByteAddressBuffer output_arrays : register(u4);

half2x2 GetMatval_mat2x2_OnUniformCompatible(UniformCompatible obj) {
    return half2x2(obj.val_mat2x2__0, obj.val_mat2x2__1);
}

void SetMatval_mat2x2_OnUniformCompatible(UniformCompatible obj, half2x2 mat) {
    obj.val_mat2x2__0 = mat[0];
    obj.val_mat2x2__1 = mat[1];
}

void SetMatVecval_mat2x2_OnUniformCompatible(UniformCompatible obj, half2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.val_mat2x2__0 = vec; break; }
    case 1: { obj.val_mat2x2__1 = vec; break; }
    }
}

void SetMatScalarval_mat2x2_OnUniformCompatible(UniformCompatible obj, half scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.val_mat2x2__0[vec_idx] = scalar; break; }
    case 1: { obj.val_mat2x2__1[vec_idx] = scalar; break; }
    }
}

half3x2 GetMatval_mat3x2_OnUniformCompatible(UniformCompatible obj) {
    return half3x2(obj.val_mat3x2__0, obj.val_mat3x2__1, obj.val_mat3x2__2);
}

void SetMatval_mat3x2_OnUniformCompatible(UniformCompatible obj, half3x2 mat) {
    obj.val_mat3x2__0 = mat[0];
    obj.val_mat3x2__1 = mat[1];
    obj.val_mat3x2__2 = mat[2];
}

void SetMatVecval_mat3x2_OnUniformCompatible(UniformCompatible obj, half2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.val_mat3x2__0 = vec; break; }
    case 1: { obj.val_mat3x2__1 = vec; break; }
    case 2: { obj.val_mat3x2__2 = vec; break; }
    }
}

void SetMatScalarval_mat3x2_OnUniformCompatible(UniformCompatible obj, half scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.val_mat3x2__0[vec_idx] = scalar; break; }
    case 1: { obj.val_mat3x2__1[vec_idx] = scalar; break; }
    case 2: { obj.val_mat3x2__2[vec_idx] = scalar; break; }
    }
}

half4x2 GetMatval_mat4x2_OnUniformCompatible(UniformCompatible obj) {
    return half4x2(obj.val_mat4x2__0, obj.val_mat4x2__1, obj.val_mat4x2__2, obj.val_mat4x2__3);
}

void SetMatval_mat4x2_OnUniformCompatible(UniformCompatible obj, half4x2 mat) {
    obj.val_mat4x2__0 = mat[0];
    obj.val_mat4x2__1 = mat[1];
    obj.val_mat4x2__2 = mat[2];
    obj.val_mat4x2__3 = mat[3];
}

void SetMatVecval_mat4x2_OnUniformCompatible(UniformCompatible obj, half2 vec, uint mat_idx) {
    switch(mat_idx) {
    case 0: { obj.val_mat4x2__0 = vec; break; }
    case 1: { obj.val_mat4x2__1 = vec; break; }
    case 2: { obj.val_mat4x2__2 = vec; break; }
    case 3: { obj.val_mat4x2__3 = vec; break; }
    }
}

void SetMatScalarval_mat4x2_OnUniformCompatible(UniformCompatible obj, half scalar, uint mat_idx, uint vec_idx) {
    switch(mat_idx) {
    case 0: { obj.val_mat4x2__0[vec_idx] = scalar; break; }
    case 1: { obj.val_mat4x2__1[vec_idx] = scalar; break; }
    case 2: { obj.val_mat4x2__2[vec_idx] = scalar; break; }
    case 3: { obj.val_mat4x2__3[vec_idx] = scalar; break; }
    }
}

typedef half ret_Constructarray2_half_[2];
ret_Constructarray2_half_ Constructarray2_half_(half arg0, half arg1) {
    half ret[2] = { arg0, arg1 };
    return ret;
}

half f16_function(half x)
{
    LayoutTest l = (LayoutTest)0;
    half val = 15.203125h;

    half phony = private_variable;
    half _e6 = val;
    val = (_e6 + -33344.0h);
    half _e8 = val;
    half _e11 = val;
    val = (_e11 + (_e8 + 5.0h));
    float _e15 = input_uniform.val_f32_;
    half _e16 = val;
    half _e20 = val;
    val = (_e20 + half((_e15 + float(_e16))));
    half _e24 = input_uniform.val_f16_;
    half _e27 = val;
    val = (_e27 + (_e24).xxx.z);
    output.Store(4, asuint(int(65504)));
    output.Store(4, asuint(int(-65504)));
    output.Store(0, asuint(65504u));
    output.Store(0, asuint(0u));
    output.Store(8, asuint(65504.0));
    output.Store(8, asuint(-65504.0));
    half _e51 = input_uniform.val_f16_;
    half _e54 = input_storage.Load<half>(12);
    output.Store(12, (_e51 + _e54));
    half2 _e60 = input_uniform.val_f16_2_;
    half2 _e63 = input_storage.Load<half2>(16);
    output.Store(16, (_e60 + _e63));
    half3 _e69 = input_uniform.val_f16_3_;
    half3 _e72 = input_storage.Load<half3>(24);
    output.Store(24, (_e69 + _e72));
    half4 _e78 = input_uniform.val_f16_4_;
    half4 _e81 = input_storage.Load<half4>(32);
    output.Store(32, (_e78 + _e81));
    half2x2 _e87 = GetMatval_mat2x2_OnUniformCompatible(input_uniform);
    half2x2 _e90 = half2x2(input_storage.Load<half2>(44+0), input_storage.Load<half2>(44+4));
    {
        half2x2 _value2 = (_e87 + _e90);
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half2x3 _e96 = input_uniform.val_mat2x3_;
    half2x3 _e99 = half2x3(input_storage.Load<half3>(56+0), input_storage.Load<half3>(56+8));
    {
        half2x3 _value2 = (_e96 + _e99);
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half2x4 _e105 = input_uniform.val_mat2x4_;
    half2x4 _e108 = half2x4(input_storage.Load<half4>(72+0), input_storage.Load<half4>(72+8));
    {
        half2x4 _value2 = (_e105 + _e108);
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half3x2 _e114 = GetMatval_mat3x2_OnUniformCompatible(input_uniform);
    half3x2 _e117 = half3x2(input_storage.Load<half2>(88+0), input_storage.Load<half2>(88+4), input_storage.Load<half2>(88+8));
    {
        half3x2 _value2 = (_e114 + _e117);
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3x3 _e123 = input_uniform.val_mat3x3_;
    half3x3 _e126 = half3x3(input_storage.Load<half3>(104+0), input_storage.Load<half3>(104+8), input_storage.Load<half3>(104+16));
    {
        half3x3 _value2 = (_e123 + _e126);
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half3x4 _e132 = input_uniform.val_mat3x4_;
    half3x4 _e135 = half3x4(input_storage.Load<half4>(128+0), input_storage.Load<half4>(128+8), input_storage.Load<half4>(128+16));
    {
        half3x4 _value2 = (_e132 + _e135);
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half4x2 _e141 = GetMatval_mat4x2_OnUniformCompatible(input_uniform);
    half4x2 _e144 = half4x2(input_storage.Load<half2>(152+0), input_storage.Load<half2>(152+4), input_storage.Load<half2>(152+8), input_storage.Load<half2>(152+12));
    {
        half4x2 _value2 = (_e141 + _e144);
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half4x3 _e150 = input_uniform.val_mat4x3_;
    half4x3 _e153 = half4x3(input_storage.Load<half3>(168+0), input_storage.Load<half3>(168+8), input_storage.Load<half3>(168+16), input_storage.Load<half3>(168+24));
    {
        half4x3 _value2 = (_e150 + _e153);
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4x4 _e159 = input_uniform.val_mat4x4_;
    half4x4 _e162 = half4x4(input_storage.Load<half4>(200+0), input_storage.Load<half4>(200+8), input_storage.Load<half4>(200+16), input_storage.Load<half4>(200+24));
    {
        half4x4 _value2 = (_e159 + _e162);
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e168[2] = Constructarray2_half_(input_arrays.Load<half>(0+0), input_arrays.Load<half>(0+2));
    {
        half _value2[2] = _e168;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    half _e169 = val;
    half _e171 = val;
    val = (_e171 + abs(_e169));
    half _e173 = val;
    half _e174 = val;
    half _e175 = val;
    half _e177 = val;
    val = (_e177 + clamp(_e173, _e174, _e175));
    half _e179 = val;
    half _e181 = val;
    half _e184 = val;
    val = (_e184 + dot((_e179).xx, (_e181).xx));
    half _e186 = val;
    half _e187 = val;
    half _e189 = val;
    val = (_e189 + max(_e186, _e187));
    half _e191 = val;
    half _e192 = val;
    half _e194 = val;
    val = (_e194 + min(_e191, _e192));
    half _e196 = val;
    half _e198 = val;
    val = (_e198 + sign(_e196));
    half _e201 = val;
    val = (_e201 + 1.0h);
    half2 _e205 = input_uniform.val_f16_2_;
    float2 float_vec2_ = float2(_e205);
    output.Store(16, half2(float_vec2_));
    half3 _e212 = input_uniform.val_f16_3_;
    float3 float_vec3_ = float3(_e212);
    output.Store(24, half3(float_vec3_));
    half4 _e219 = input_uniform.val_f16_4_;
    float4 float_vec4_ = float4(_e219);
    output.Store(32, half4(float_vec4_));
    half2x2 _e228 = GetMatval_mat2x2_OnUniformCompatible(input_uniform);
    {
        half2x2 _value2 = half2x2(float2x2(_e228));
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half2x3 _e235 = input_uniform.val_mat2x3_;
    {
        half2x3 _value2 = half2x3(float2x3(_e235));
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half2x4 _e242 = input_uniform.val_mat2x4_;
    {
        half2x4 _value2 = half2x4(float2x4(_e242));
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half3x2 _e249 = GetMatval_mat3x2_OnUniformCompatible(input_uniform);
    {
        half3x2 _value2 = half3x2(float3x2(_e249));
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3x3 _e256 = input_uniform.val_mat3x3_;
    {
        half3x3 _value2 = half3x3(float3x3(_e256));
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half3x4 _e263 = input_uniform.val_mat3x4_;
    {
        half3x4 _value2 = half3x4(float3x4(_e263));
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half4x2 _e270 = GetMatval_mat4x2_OnUniformCompatible(input_uniform);
    {
        half4x2 _value2 = half4x2(float4x2(_e270));
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half4x3 _e277 = input_uniform.val_mat4x3_;
    {
        half4x3 _value2 = half4x3(float4x3(_e277));
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4x4 _e284 = input_uniform.val_mat4x4_;
    {
        half4x4 _value2 = half4x4(float4x4(_e284));
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e287 = val;
    return _e287;
}

[numthreads(1, 1, 1)]
void main()
{
    const half _e3 = f16_function(2.0h);
    output.Store(40, _e3);
    return;
}

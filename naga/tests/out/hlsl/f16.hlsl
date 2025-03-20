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
    half val = 15.203125h;

    half _e4 = val;
    val = (_e4 + -33344.0h);
    half _e6 = val;
    half _e9 = val;
    val = (_e9 + (_e6 + 5.0h));
    float _e13 = input_uniform.val_f32_;
    half _e14 = val;
    half _e18 = val;
    val = (_e18 + half((_e13 + float(_e14))));
    half _e22 = input_uniform.val_f16_;
    half _e25 = val;
    val = (_e25 + (_e22).xxx.z);
    half _e31 = input_uniform.val_f16_;
    half _e34 = input_storage.Load<half>(12);
    output.Store(12, (_e31 + _e34));
    half2 _e40 = input_uniform.val_f16_2_;
    half2 _e43 = input_storage.Load<half2>(16);
    output.Store(16, (_e40 + _e43));
    half3 _e49 = input_uniform.val_f16_3_;
    half3 _e52 = input_storage.Load<half3>(24);
    output.Store(24, (_e49 + _e52));
    half4 _e58 = input_uniform.val_f16_4_;
    half4 _e61 = input_storage.Load<half4>(32);
    output.Store(32, (_e58 + _e61));
    half2x2 _e67 = GetMatval_mat2x2_OnUniformCompatible(input_uniform);
    half2x2 _e70 = half2x2(input_storage.Load<half2>(44+0), input_storage.Load<half2>(44+4));
    {
        half2x2 _value2 = (_e67 + _e70);
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half2x3 _e76 = input_uniform.val_mat2x3_;
    half2x3 _e79 = half2x3(input_storage.Load<half3>(56+0), input_storage.Load<half3>(56+8));
    {
        half2x3 _value2 = (_e76 + _e79);
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half2x4 _e85 = input_uniform.val_mat2x4_;
    half2x4 _e88 = half2x4(input_storage.Load<half4>(72+0), input_storage.Load<half4>(72+8));
    {
        half2x4 _value2 = (_e85 + _e88);
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half3x2 _e94 = GetMatval_mat3x2_OnUniformCompatible(input_uniform);
    half3x2 _e97 = half3x2(input_storage.Load<half2>(88+0), input_storage.Load<half2>(88+4), input_storage.Load<half2>(88+8));
    {
        half3x2 _value2 = (_e94 + _e97);
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3x3 _e103 = input_uniform.val_mat3x3_;
    half3x3 _e106 = half3x3(input_storage.Load<half3>(104+0), input_storage.Load<half3>(104+8), input_storage.Load<half3>(104+16));
    {
        half3x3 _value2 = (_e103 + _e106);
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half3x4 _e112 = input_uniform.val_mat3x4_;
    half3x4 _e115 = half3x4(input_storage.Load<half4>(128+0), input_storage.Load<half4>(128+8), input_storage.Load<half4>(128+16));
    {
        half3x4 _value2 = (_e112 + _e115);
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half4x2 _e121 = GetMatval_mat4x2_OnUniformCompatible(input_uniform);
    half4x2 _e124 = half4x2(input_storage.Load<half2>(152+0), input_storage.Load<half2>(152+4), input_storage.Load<half2>(152+8), input_storage.Load<half2>(152+12));
    {
        half4x2 _value2 = (_e121 + _e124);
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half4x3 _e130 = input_uniform.val_mat4x3_;
    half4x3 _e133 = half4x3(input_storage.Load<half3>(168+0), input_storage.Load<half3>(168+8), input_storage.Load<half3>(168+16), input_storage.Load<half3>(168+24));
    {
        half4x3 _value2 = (_e130 + _e133);
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4x4 _e139 = input_uniform.val_mat4x4_;
    half4x4 _e142 = half4x4(input_storage.Load<half4>(200+0), input_storage.Load<half4>(200+8), input_storage.Load<half4>(200+16), input_storage.Load<half4>(200+24));
    {
        half4x4 _value2 = (_e139 + _e142);
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e148[2] = Constructarray2_half_(input_arrays.Load<half>(0+0), input_arrays.Load<half>(0+2));
    {
        half _value2[2] = _e148;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    half _e149 = val;
    half _e151 = val;
    val = (_e151 + abs(_e149));
    half _e153 = val;
    half _e154 = val;
    half _e155 = val;
    half _e157 = val;
    val = (_e157 + clamp(_e153, _e154, _e155));
    half _e159 = val;
    half _e161 = val;
    half _e164 = val;
    val = (_e164 + dot((_e159).xx, (_e161).xx));
    half _e166 = val;
    half _e167 = val;
    half _e169 = val;
    val = (_e169 + max(_e166, _e167));
    half _e171 = val;
    half _e172 = val;
    half _e174 = val;
    val = (_e174 + min(_e171, _e172));
    half _e176 = val;
    half _e178 = val;
    val = (_e178 + sign(_e176));
    half _e181 = val;
    val = (_e181 + 1.0h);
    half2 _e185 = input_uniform.val_f16_2_;
    float2 float_vec2_ = float2(_e185);
    output.Store(16, half2(float_vec2_));
    half3 _e192 = input_uniform.val_f16_3_;
    float3 float_vec3_ = float3(_e192);
    output.Store(24, half3(float_vec3_));
    half4 _e199 = input_uniform.val_f16_4_;
    float4 float_vec4_ = float4(_e199);
    output.Store(32, half4(float_vec4_));
    half2x2 _e208 = GetMatval_mat2x2_OnUniformCompatible(input_uniform);
    {
        half2x2 _value2 = half2x2(float2x2(_e208));
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half2x3 _e215 = input_uniform.val_mat2x3_;
    {
        half2x3 _value2 = half2x3(float2x3(_e215));
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half2x4 _e222 = input_uniform.val_mat2x4_;
    {
        half2x4 _value2 = half2x4(float2x4(_e222));
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half3x2 _e229 = GetMatval_mat3x2_OnUniformCompatible(input_uniform);
    {
        half3x2 _value2 = half3x2(float3x2(_e229));
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3x3 _e236 = input_uniform.val_mat3x3_;
    {
        half3x3 _value2 = half3x3(float3x3(_e236));
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half3x4 _e243 = input_uniform.val_mat3x4_;
    {
        half3x4 _value2 = half3x4(float3x4(_e243));
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half4x2 _e250 = GetMatval_mat4x2_OnUniformCompatible(input_uniform);
    {
        half4x2 _value2 = half4x2(float4x2(_e250));
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half4x3 _e257 = input_uniform.val_mat4x3_;
    {
        half4x3 _value2 = half4x3(float4x3(_e257));
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4x4 _e264 = input_uniform.val_mat4x4_;
    {
        half4x4 _value2 = half4x4(float4x4(_e264));
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e267 = val;
    return _e267;
}

[numthreads(1, 1, 1)]
void main()
{
    const half _e3 = f16_function(2.0h);
    output.Store(40, _e3);
    return;
}

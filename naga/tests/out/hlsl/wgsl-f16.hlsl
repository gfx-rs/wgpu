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

LayoutTest ZeroValueLayoutTest() {
    return (LayoutTest)0;
}

half2x2 GetMatval_mat2x2_OnUniformCompatible(UniformCompatible obj) {
    return half2x2(obj.val_mat2x2__0, obj.val_mat2x2__1);
}

half3x2 GetMatval_mat3x2_OnUniformCompatible(UniformCompatible obj) {
    return half3x2(obj.val_mat3x2__0, obj.val_mat3x2__1, obj.val_mat3x2__2);
}

half4x2 GetMatval_mat4x2_OnUniformCompatible(UniformCompatible obj) {
    return half4x2(obj.val_mat4x2__0, obj.val_mat4x2__1, obj.val_mat4x2__2, obj.val_mat4x2__3);
}

typedef half ret_Constructarray2_half_[2];
ret_Constructarray2_half_ Constructarray2_half_(half arg0, half arg1) {
    half ret[2] = { arg0, arg1 };
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    half x = (half)0;
    LayoutTest l = (LayoutTest)0;
    half val = (half)0;
    half local = (half)0;

    x = 2.0h;
    l = ZeroValueLayoutTest();
    val = 15.203125h;
    half _e3 = val;
    val = (_e3 + -33344.0h);
    half _e6 = val;
    half _e7 = val;
    val = (_e6 + (_e7 + 5.0h));
    half _e11 = val;
    float _e14 = input_uniform.val_f32_;
    half _e15 = val;
    val = (_e11 + half((_e14 + float(_e15))));
    half _e20 = val;
    half _e23 = input_uniform.val_f16_;
    val = (_e20 + (_e23).xxx.z);
    output.Store(4, asuint(int(65504)));
    output.Store(4, asuint(int(-65504)));
    output.Store(0, asuint(65504u));
    output.Store(0, asuint(0u));
    output.Store(8, asuint(65504.0));
    output.Store(8, asuint(-65504.0));
    half _e43 = input_uniform.val_f16_;
    half _e46 = input_storage.Load<half>(12);
    output.Store(12, (_e43 + _e46));
    half2 _e52 = input_uniform.val_f16_2_;
    half2 _e55 = input_storage.Load<half2>(16);
    output.Store(16, (_e52 + _e55));
    half3 _e61 = input_uniform.val_f16_3_;
    half3 _e64 = input_storage.Load<half3>(24);
    output.Store(24, (_e61 + _e64));
    half4 _e70 = input_uniform.val_f16_4_;
    half4 _e73 = input_storage.Load<half4>(32);
    output.Store(32, (_e70 + _e73));
    half2x2 _e79 = GetMatval_mat2x2_OnUniformCompatible(input_uniform);
    half2x2 _e82 = half2x2(input_storage.Load<half2>(44+0), input_storage.Load<half2>(44+4));
    {
        half2x2 _value2 = (_e79 + _e82);
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half2x3 _e88 = input_uniform.val_mat2x3_;
    half2x3 _e91 = half2x3(input_storage.Load<half3>(56+0), input_storage.Load<half3>(56+8));
    {
        half2x3 _value2 = (_e88 + _e91);
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half2x4 _e97 = input_uniform.val_mat2x4_;
    half2x4 _e100 = half2x4(input_storage.Load<half4>(72+0), input_storage.Load<half4>(72+8));
    {
        half2x4 _value2 = (_e97 + _e100);
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half3x2 _e106 = GetMatval_mat3x2_OnUniformCompatible(input_uniform);
    half3x2 _e109 = half3x2(input_storage.Load<half2>(88+0), input_storage.Load<half2>(88+4), input_storage.Load<half2>(88+8));
    {
        half3x2 _value2 = (_e106 + _e109);
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3x3 _e115 = input_uniform.val_mat3x3_;
    half3x3 _e118 = half3x3(input_storage.Load<half3>(104+0), input_storage.Load<half3>(104+8), input_storage.Load<half3>(104+16));
    {
        half3x3 _value2 = (_e115 + _e118);
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half3x4 _e124 = input_uniform.val_mat3x4_;
    half3x4 _e127 = half3x4(input_storage.Load<half4>(128+0), input_storage.Load<half4>(128+8), input_storage.Load<half4>(128+16));
    {
        half3x4 _value2 = (_e124 + _e127);
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half4x2 _e133 = GetMatval_mat4x2_OnUniformCompatible(input_uniform);
    half4x2 _e136 = half4x2(input_storage.Load<half2>(152+0), input_storage.Load<half2>(152+4), input_storage.Load<half2>(152+8), input_storage.Load<half2>(152+12));
    {
        half4x2 _value2 = (_e133 + _e136);
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half4x3 _e142 = input_uniform.val_mat4x3_;
    half4x3 _e145 = half4x3(input_storage.Load<half3>(168+0), input_storage.Load<half3>(168+8), input_storage.Load<half3>(168+16), input_storage.Load<half3>(168+24));
    {
        half4x3 _value2 = (_e142 + _e145);
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4x4 _e151 = input_uniform.val_mat4x4_;
    half4x4 _e154 = half4x4(input_storage.Load<half4>(200+0), input_storage.Load<half4>(200+8), input_storage.Load<half4>(200+16), input_storage.Load<half4>(200+24));
    {
        half4x4 _value2 = (_e151 + _e154);
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e160[2] = Constructarray2_half_(input_arrays.Load<half>(0+0), input_arrays.Load<half>(0+2));
    {
        half _value2[2] = _e160;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    half _e161 = val;
    half _e162 = val;
    val = (_e161 + abs(_e162));
    half _e165 = val;
    half _e166 = val;
    half _e167 = val;
    half _e168 = val;
    val = (_e165 + clamp(_e166, _e167, _e168));
    half _e171 = val;
    half _e172 = val;
    half _e174 = val;
    val = (_e171 + dot((_e172).xx, (_e174).xx));
    half _e178 = val;
    half _e179 = val;
    half _e180 = val;
    val = (_e178 + max(_e179, _e180));
    half _e183 = val;
    half _e184 = val;
    half _e185 = val;
    val = (_e183 + min(_e184, _e185));
    half _e188 = val;
    half _e189 = val;
    val = (_e188 + half(sign(_e189)));
    half _e192 = val;
    val = (_e192 + 1.0h);
    half2 _e197 = input_uniform.val_f16_2_;
    output.Store(16, half2(float2(_e197)));
    half3 _e204 = input_uniform.val_f16_3_;
    output.Store(24, half3(float3(_e204)));
    half4 _e211 = input_uniform.val_f16_4_;
    output.Store(32, half4(float4(_e211)));
    half2x2 _e220 = GetMatval_mat2x2_OnUniformCompatible(input_uniform);
    {
        half2x2 _value2 = half2x2(float2x2(_e220));
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half2x3 _e227 = input_uniform.val_mat2x3_;
    {
        half2x3 _value2 = half2x3(float2x3(_e227));
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half2x4 _e234 = input_uniform.val_mat2x4_;
    {
        half2x4 _value2 = half2x4(float2x4(_e234));
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half3x2 _e241 = GetMatval_mat3x2_OnUniformCompatible(input_uniform);
    {
        half3x2 _value2 = half3x2(float3x2(_e241));
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3x3 _e248 = input_uniform.val_mat3x3_;
    {
        half3x3 _value2 = half3x3(float3x3(_e248));
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half3x4 _e255 = input_uniform.val_mat3x4_;
    {
        half3x4 _value2 = half3x4(float3x4(_e255));
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half4x2 _e262 = GetMatval_mat4x2_OnUniformCompatible(input_uniform);
    {
        half4x2 _value2 = half4x2(float4x2(_e262));
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half4x3 _e269 = input_uniform.val_mat4x3_;
    {
        half4x3 _value2 = half4x3(float4x3(_e269));
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4x4 _e276 = input_uniform.val_mat4x4_;
    {
        half4x4 _value2 = half4x4(float4x4(_e276));
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e279 = val;
    local = _e279;
    half _e281 = local;
    output.Store(40, _e281);
    return;
}

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
    uint16_t _pad4_0;
    half2 val_f16_2_;
    int _pad5_0;
    half3 val_f16_3_;
    uint16_t _pad6_0;
    half4 val_f16_4_;
    half final_value;
    uint16_t _pad8_0;
    row_major half2x2 val_mat2x2_;
    int _pad9_0;
    row_major half2x3 val_mat2x3_;
    uint16_t _pad10_0;
    row_major half2x4 val_mat2x4_;
    row_major half3x2 val_mat3x2_;
    int _pad12_0;
    row_major half3x3 val_mat3x3_;
    uint16_t _pad13_0;
    row_major half3x4 val_mat3x4_;
    row_major half4x2 val_mat4x2_;
    row_major half4x3 val_mat4x3_;
    uint16_t _pad16_0;
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
    uint16_t _pad5_0;
    uint larger;
};

struct UniformCompatible_std140_ {
    uint val_u32_;
    int val_i32_;
    float val_f32_;
    half val_f16_;
    uint16_t _pad4_0;
    half2 val_f16_2_;
    int _pad5_0;
    half3 val_f16_3_;
    uint16_t _pad6_0;
    half4 val_f16_4_;
    half final_value;
    uint16_t _pad8_0;
    half2 val_mat2x2_0_;
    half2 val_mat2x2_1_;
    int _pad10_0;
    half3 val_mat2x3_0_;
    uint16_t _pad11_0;
    half3 val_mat2x3_1_;
    uint16_t _pad12_0;
    half4 val_mat2x4_0_;
    half4 val_mat2x4_1_;
    half2 val_mat3x2_0_;
    half2 val_mat3x2_1_;
    half2 val_mat3x2_2_;
    int _pad17_0;
    half3 val_mat3x3_0_;
    uint16_t _pad18_0;
    half3 val_mat3x3_1_;
    uint16_t _pad19_0;
    half3 val_mat3x3_2_;
    uint16_t _pad20_0;
    half4 val_mat3x4_0_;
    half4 val_mat3x4_1_;
    half4 val_mat3x4_2_;
    half2 val_mat4x2_0_;
    half2 val_mat4x2_1_;
    half2 val_mat4x2_2_;
    half2 val_mat4x2_3_;
    half3 val_mat4x3_0_;
    uint16_t _pad28_0;
    half3 val_mat4x3_1_;
    uint16_t _pad29_0;
    half3 val_mat4x3_2_;
    uint16_t _pad30_0;
    half3 val_mat4x3_3_;
    uint16_t _pad31_0;
    half4 val_mat4x4_0_;
    half4 val_mat4x4_1_;
    half4 val_mat4x4_2_;
    half4 val_mat4x4_3_;
};

static const half constant_variable = 15.203125h;

static half private_variable = 1.0h;
cbuffer input_uniform : register(b0) { UniformCompatible_std140_ input_uniform; }
ByteAddressBuffer input_storage : register(t1);
ByteAddressBuffer input_arrays : register(t2);
RWByteAddressBuffer output : register(u3);
RWByteAddressBuffer output_arrays : register(u4);

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
    half2 _e79 = input_uniform.val_mat2x2_0_;
    half2 _e81 = input_uniform.val_mat2x2_1_;
    half2x2 _e85 = half2x2(input_storage.Load<half2>(44+0), input_storage.Load<half2>(44+4));
    {
        half2x2 _value2 = (half2x2(_e79, _e81) + _e85);
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half3 _e91 = input_uniform.val_mat2x3_0_;
    half3 _e93 = input_uniform.val_mat2x3_1_;
    half2x3 _e97 = half2x3(input_storage.Load<half3>(56+0), input_storage.Load<half3>(56+8));
    {
        half2x3 _value2 = (half2x3(_e91, _e93) + _e97);
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half4 _e103 = input_uniform.val_mat2x4_0_;
    half4 _e105 = input_uniform.val_mat2x4_1_;
    half2x4 _e109 = half2x4(input_storage.Load<half4>(72+0), input_storage.Load<half4>(72+8));
    {
        half2x4 _value2 = (half2x4(_e103, _e105) + _e109);
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half2 _e115 = input_uniform.val_mat3x2_0_;
    half2 _e117 = input_uniform.val_mat3x2_1_;
    half2 _e119 = input_uniform.val_mat3x2_2_;
    half3x2 _e123 = half3x2(input_storage.Load<half2>(88+0), input_storage.Load<half2>(88+4), input_storage.Load<half2>(88+8));
    {
        half3x2 _value2 = (half3x2(_e115, _e117, _e119) + _e123);
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3 _e129 = input_uniform.val_mat3x3_0_;
    half3 _e131 = input_uniform.val_mat3x3_1_;
    half3 _e133 = input_uniform.val_mat3x3_2_;
    half3x3 _e137 = half3x3(input_storage.Load<half3>(104+0), input_storage.Load<half3>(104+8), input_storage.Load<half3>(104+16));
    {
        half3x3 _value2 = (half3x3(_e129, _e131, _e133) + _e137);
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half4 _e143 = input_uniform.val_mat3x4_0_;
    half4 _e145 = input_uniform.val_mat3x4_1_;
    half4 _e147 = input_uniform.val_mat3x4_2_;
    half3x4 _e151 = half3x4(input_storage.Load<half4>(128+0), input_storage.Load<half4>(128+8), input_storage.Load<half4>(128+16));
    {
        half3x4 _value2 = (half3x4(_e143, _e145, _e147) + _e151);
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half2 _e157 = input_uniform.val_mat4x2_0_;
    half2 _e159 = input_uniform.val_mat4x2_1_;
    half2 _e161 = input_uniform.val_mat4x2_2_;
    half2 _e163 = input_uniform.val_mat4x2_3_;
    half4x2 _e167 = half4x2(input_storage.Load<half2>(152+0), input_storage.Load<half2>(152+4), input_storage.Load<half2>(152+8), input_storage.Load<half2>(152+12));
    {
        half4x2 _value2 = (half4x2(_e157, _e159, _e161, _e163) + _e167);
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half3 _e173 = input_uniform.val_mat4x3_0_;
    half3 _e175 = input_uniform.val_mat4x3_1_;
    half3 _e177 = input_uniform.val_mat4x3_2_;
    half3 _e179 = input_uniform.val_mat4x3_3_;
    half4x3 _e183 = half4x3(input_storage.Load<half3>(168+0), input_storage.Load<half3>(168+8), input_storage.Load<half3>(168+16), input_storage.Load<half3>(168+24));
    {
        half4x3 _value2 = (half4x3(_e173, _e175, _e177, _e179) + _e183);
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4 _e189 = input_uniform.val_mat4x4_0_;
    half4 _e191 = input_uniform.val_mat4x4_1_;
    half4 _e193 = input_uniform.val_mat4x4_2_;
    half4 _e195 = input_uniform.val_mat4x4_3_;
    half4x4 _e199 = half4x4(input_storage.Load<half4>(200+0), input_storage.Load<half4>(200+8), input_storage.Load<half4>(200+16), input_storage.Load<half4>(200+24));
    {
        half4x4 _value2 = (half4x4(_e189, _e191, _e193, _e195) + _e199);
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e205[2] = Constructarray2_half_(input_arrays.Load<half>(0+0), input_arrays.Load<half>(0+2));
    {
        half _value2[2] = _e205;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    half _e206 = val;
    half _e207 = val;
    val = (_e206 + abs(_e207));
    half _e210 = val;
    half _e211 = val;
    half _e212 = val;
    half _e213 = val;
    val = (_e210 + clamp(_e211, _e212, _e213));
    half _e216 = val;
    half _e217 = val;
    half _e219 = val;
    val = (_e216 + dot((_e217).xx, (_e219).xx));
    half _e223 = val;
    half _e224 = val;
    half _e225 = val;
    val = (_e223 + max(_e224, _e225));
    half _e228 = val;
    half _e229 = val;
    half _e230 = val;
    val = (_e228 + min(_e229, _e230));
    half _e233 = val;
    half _e234 = val;
    val = (_e233 + half(sign(_e234)));
    half _e237 = val;
    val = (_e237 + 1.0h);
    half2 _e242 = input_uniform.val_f16_2_;
    float2 float_vec2_ = float2(_e242);
    output.Store(16, half2(float_vec2_));
    half3 _e249 = input_uniform.val_f16_3_;
    float3 float_vec3_ = float3(_e249);
    output.Store(24, half3(float_vec3_));
    half4 _e256 = input_uniform.val_f16_4_;
    float4 float_vec4_ = float4(_e256);
    output.Store(32, half4(float_vec4_));
    half2 _e265 = input_uniform.val_mat2x2_0_;
    half2 _e267 = input_uniform.val_mat2x2_1_;
    {
        half2x2 _value2 = half2x2(float2x2(half2x2(_e265, _e267)));
        output.Store(44+0, _value2[0]);
        output.Store(44+4, _value2[1]);
    }
    half3 _e275 = input_uniform.val_mat2x3_0_;
    half3 _e277 = input_uniform.val_mat2x3_1_;
    {
        half2x3 _value2 = half2x3(float2x3(half2x3(_e275, _e277)));
        output.Store(56+0, _value2[0]);
        output.Store(56+8, _value2[1]);
    }
    half4 _e285 = input_uniform.val_mat2x4_0_;
    half4 _e287 = input_uniform.val_mat2x4_1_;
    {
        half2x4 _value2 = half2x4(float2x4(half2x4(_e285, _e287)));
        output.Store(72+0, _value2[0]);
        output.Store(72+8, _value2[1]);
    }
    half2 _e295 = input_uniform.val_mat3x2_0_;
    half2 _e297 = input_uniform.val_mat3x2_1_;
    half2 _e299 = input_uniform.val_mat3x2_2_;
    {
        half3x2 _value2 = half3x2(float3x2(half3x2(_e295, _e297, _e299)));
        output.Store(88+0, _value2[0]);
        output.Store(88+4, _value2[1]);
        output.Store(88+8, _value2[2]);
    }
    half3 _e307 = input_uniform.val_mat3x3_0_;
    half3 _e309 = input_uniform.val_mat3x3_1_;
    half3 _e311 = input_uniform.val_mat3x3_2_;
    {
        half3x3 _value2 = half3x3(float3x3(half3x3(_e307, _e309, _e311)));
        output.Store(104+0, _value2[0]);
        output.Store(104+8, _value2[1]);
        output.Store(104+16, _value2[2]);
    }
    half4 _e319 = input_uniform.val_mat3x4_0_;
    half4 _e321 = input_uniform.val_mat3x4_1_;
    half4 _e323 = input_uniform.val_mat3x4_2_;
    {
        half3x4 _value2 = half3x4(float3x4(half3x4(_e319, _e321, _e323)));
        output.Store(128+0, _value2[0]);
        output.Store(128+8, _value2[1]);
        output.Store(128+16, _value2[2]);
    }
    half2 _e331 = input_uniform.val_mat4x2_0_;
    half2 _e333 = input_uniform.val_mat4x2_1_;
    half2 _e335 = input_uniform.val_mat4x2_2_;
    half2 _e337 = input_uniform.val_mat4x2_3_;
    {
        half4x2 _value2 = half4x2(float4x2(half4x2(_e331, _e333, _e335, _e337)));
        output.Store(152+0, _value2[0]);
        output.Store(152+4, _value2[1]);
        output.Store(152+8, _value2[2]);
        output.Store(152+12, _value2[3]);
    }
    half3 _e345 = input_uniform.val_mat4x3_0_;
    half3 _e347 = input_uniform.val_mat4x3_1_;
    half3 _e349 = input_uniform.val_mat4x3_2_;
    half3 _e351 = input_uniform.val_mat4x3_3_;
    {
        half4x3 _value2 = half4x3(float4x3(half4x3(_e345, _e347, _e349, _e351)));
        output.Store(168+0, _value2[0]);
        output.Store(168+8, _value2[1]);
        output.Store(168+16, _value2[2]);
        output.Store(168+24, _value2[3]);
    }
    half4 _e359 = input_uniform.val_mat4x4_0_;
    half4 _e361 = input_uniform.val_mat4x4_1_;
    half4 _e363 = input_uniform.val_mat4x4_2_;
    half4 _e365 = input_uniform.val_mat4x4_3_;
    {
        half4x4 _value2 = half4x4(float4x4(half4x4(_e359, _e361, _e363, _e365)));
        output.Store(200+0, _value2[0]);
        output.Store(200+8, _value2[1]);
        output.Store(200+16, _value2[2]);
        output.Store(200+24, _value2[3]);
    }
    half _e369 = val;
    return _e369;
}

[numthreads(1, 1, 1)]
void main()
{
    const half _e3 = f16_function(2.0h);
    output.Store(40, _e3);
    return;
}

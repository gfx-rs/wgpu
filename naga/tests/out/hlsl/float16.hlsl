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
    int _end_pad_0;
};

struct StorageCompatible {
    half val_f16_array_2_[2];
    half val_f16_array_2_1[2];
};

static const half constant_variable = 15.203125h;

static half private_variable = 1.0h;
cbuffer input_uniform : register(b0) { UniformCompatible input_uniform; }
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
    half val = 15.203125h;

    half _expr6 = val;
    val = (_expr6 + (1.0h - 33344.0h));
    half _expr8 = val;
    half _expr11 = val;
    val = (_expr11 + (_expr8 + 5.0h));
    float _expr15 = input_uniform.val_f32_;
    half _expr16 = val;
    half _expr20 = val;
    val = (_expr20 + half((_expr15 + float(_expr16))));
    half _expr24 = input_uniform.val_f16_;
    half _expr27 = val;
    val = (_expr27 + (_expr24).xxx.z);
    half _expr33 = input_uniform.val_f16_;
    half _expr36 = input_storage.Load<half>(12);
    output.Store(12, (_expr33 + _expr36));
    half2 _expr42 = input_uniform.val_f16_2_;
    half2 _expr45 = input_storage.Load<half2>(16);
    output.Store(16, (_expr42 + _expr45));
    half3 _expr51 = input_uniform.val_f16_3_;
    half3 _expr54 = input_storage.Load<half3>(24);
    output.Store(24, (_expr51 + _expr54));
    half4 _expr60 = input_uniform.val_f16_4_;
    half4 _expr63 = input_storage.Load<half4>(32);
    output.Store(32, (_expr60 + _expr63));
    half _expr69[2] = Constructarray2_half_(input_arrays.Load<half>(0+0), input_arrays.Load<half>(0+2));
    {
        half _value2[2] = _expr69;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    half _expr70 = val;
    half _expr72 = val;
    val = (_expr72 + abs(_expr70));
    half _expr74 = val;
    half _expr75 = val;
    half _expr76 = val;
    half _expr78 = val;
    val = (_expr78 + clamp(_expr74, _expr75, _expr76));
    half _expr80 = val;
    half _expr82 = val;
    half _expr85 = val;
    val = (_expr85 + dot((_expr80).xx, (_expr82).xx));
    half _expr87 = val;
    half _expr88 = val;
    half _expr90 = val;
    val = (_expr90 + max(_expr87, _expr88));
    half _expr92 = val;
    half _expr93 = val;
    half _expr95 = val;
    val = (_expr95 + min(_expr92, _expr93));
    half _expr97 = val;
    half _expr99 = val;
    val = (_expr99 + sign(_expr97));
    return 1.0h;
}

[numthreads(1, 1, 1)]
void main()
{
    const half _e3 = f16_function(2.0h);
    output.Store(40, _e3);
    return;
}

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
    int _pad3_0;
    uint64_t val_u64_;
    int _pad4_0;
    int _pad4_1;
    uint64_t2 val_u64_2_;
    int _pad5_0;
    int _pad5_1;
    int _pad5_2;
    int _pad5_3;
    uint64_t3 val_u64_3_;
    int _pad6_0;
    int _pad6_1;
    uint64_t4 val_u64_4_;
    int64_t val_i64_;
    int _pad8_0;
    int _pad8_1;
    int64_t2 val_i64_2_;
    int64_t3 val_i64_3_;
    int _pad10_0;
    int _pad10_1;
    int64_t4 val_i64_4_;
    uint64_t final_value;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
    int _end_pad_3;
    int _end_pad_4;
    int _end_pad_5;
};

struct StorageCompatible {
    uint64_t val_u64_array_2_[2];
    int64_t val_i64_array_2_[2];
};

static const uint64_t constant_variable = 20uL;

static int64_t private_variable = 1L;
cbuffer input_uniform : register(b0) { UniformCompatible input_uniform; }
ByteAddressBuffer input_storage : register(t1);
ByteAddressBuffer input_arrays : register(t2);
RWByteAddressBuffer output : register(u3);
RWByteAddressBuffer output_arrays : register(u4);

typedef int64_t ret_Constructarray2_int64_t_[2];
ret_Constructarray2_int64_t_ Constructarray2_int64_t_(int64_t arg0, int64_t arg1) {
    int64_t ret[2] = { arg0, arg1 };
    return ret;
}

int64_t int64_function(int64_t x)
{
    int64_t val = 20L;

    int64_t _e6 = val;
    val = (_e6 + (31L - 1002003004005006L));
    int64_t _e8 = val;
    int64_t _e11 = val;
    val = (_e11 + (_e8 + 5L));
    uint _e15 = input_uniform.val_u32_;
    int64_t _e16 = val;
    int64_t _e20 = val;
    val = (_e20 + int64_t((_e15 + uint(_e16))));
    int _e24 = input_uniform.val_i32_;
    int64_t _e25 = val;
    int64_t _e29 = val;
    val = (_e29 + int64_t((_e24 + int(_e25))));
    float _e33 = input_uniform.val_f32_;
    int64_t _e34 = val;
    int64_t _e38 = val;
    val = (_e38 + int64_t((_e33 + float(_e34))));
    int64_t _e42 = input_uniform.val_i64_;
    int64_t _e45 = val;
    val = (_e45 + (_e42).xxx.z);
    uint64_t _e49 = input_uniform.val_u64_;
    int64_t _e51 = val;
    val = (_e51 + _e49);
    uint64_t2 _e55 = input_uniform.val_u64_2_;
    int64_t _e58 = val;
    val = (_e58 + _e55.y);
    uint64_t3 _e62 = input_uniform.val_u64_3_;
    int64_t _e65 = val;
    val = (_e65 + _e62.z);
    uint64_t4 _e69 = input_uniform.val_u64_4_;
    int64_t _e72 = val;
    val = (_e72 + _e69.w);
    int64_t _e78 = input_uniform.val_i64_;
    int64_t _e81 = input_storage.Load<int64_t>(128);
    output.Store(128, (_e78 + _e81));
    int64_t2 _e87 = input_uniform.val_i64_2_;
    int64_t2 _e90 = input_storage.Load<int64_t2>(144);
    output.Store(144, (_e87 + _e90));
    int64_t3 _e96 = input_uniform.val_i64_3_;
    int64_t3 _e99 = input_storage.Load<int64_t3>(160);
    output.Store(160, (_e96 + _e99));
    int64_t4 _e105 = input_uniform.val_i64_4_;
    int64_t4 _e108 = input_storage.Load<int64_t4>(192);
    output.Store(192, (_e105 + _e108));
    int64_t _e114[2] = Constructarray2_int64_t_(input_arrays.Load<int64_t>(16+0), input_arrays.Load<int64_t>(16+8));
    {
        int64_t _value2[2] = _e114;
        output_arrays.Store(16+0, _value2[0]);
        output_arrays.Store(16+8, _value2[1]);
    }
    int64_t _e115 = val;
    int64_t _e117 = val;
    val = (_e117 + abs(_e115));
    int64_t _e119 = val;
    int64_t _e120 = val;
    int64_t _e121 = val;
    int64_t _e123 = val;
    val = (_e123 + clamp(_e119, _e120, _e121));
    int64_t _e125 = val;
    int64_t _e127 = val;
    int64_t _e130 = val;
    val = (_e130 + dot((_e125).xx, (_e127).xx));
    int64_t _e132 = val;
    int64_t _e133 = val;
    int64_t _e135 = val;
    val = (_e135 + max(_e132, _e133));
    int64_t _e137 = val;
    int64_t _e138 = val;
    int64_t _e140 = val;
    val = (_e140 + min(_e137, _e138));
    int64_t _e142 = val;
    int64_t _e144 = val;
    val = (_e144 + sign(_e142));
    int64_t _e146 = val;
    return _e146;
}

typedef uint64_t ret_Constructarray2_uint64_t_[2];
ret_Constructarray2_uint64_t_ Constructarray2_uint64_t_(uint64_t arg0, uint64_t arg1) {
    uint64_t ret[2] = { arg0, arg1 };
    return ret;
}

uint64_t uint64_function(uint64_t x_1)
{
    uint64_t val_1 = 20uL;

    uint64_t _e6 = val_1;
    val_1 = (_e6 + (31uL + 1002003004005006uL));
    uint64_t _e8 = val_1;
    uint64_t _e11 = val_1;
    val_1 = (_e11 + (_e8 + 5uL));
    uint _e15 = input_uniform.val_u32_;
    uint64_t _e16 = val_1;
    uint64_t _e20 = val_1;
    val_1 = (_e20 + uint64_t((_e15 + uint(_e16))));
    int _e24 = input_uniform.val_i32_;
    uint64_t _e25 = val_1;
    uint64_t _e29 = val_1;
    val_1 = (_e29 + uint64_t((_e24 + int(_e25))));
    float _e33 = input_uniform.val_f32_;
    uint64_t _e34 = val_1;
    uint64_t _e38 = val_1;
    val_1 = (_e38 + uint64_t((_e33 + float(_e34))));
    uint64_t _e42 = input_uniform.val_u64_;
    uint64_t _e45 = val_1;
    val_1 = (_e45 + (_e42).xxx.z);
    int64_t _e49 = input_uniform.val_i64_;
    uint64_t _e51 = val_1;
    val_1 = (_e51 + _e49);
    int64_t2 _e55 = input_uniform.val_i64_2_;
    uint64_t _e58 = val_1;
    val_1 = (_e58 + _e55.y);
    int64_t3 _e62 = input_uniform.val_i64_3_;
    uint64_t _e65 = val_1;
    val_1 = (_e65 + _e62.z);
    int64_t4 _e69 = input_uniform.val_i64_4_;
    uint64_t _e72 = val_1;
    val_1 = (_e72 + _e69.w);
    uint64_t _e78 = input_uniform.val_u64_;
    uint64_t _e81 = input_storage.Load<uint64_t>(16);
    output.Store(16, (_e78 + _e81));
    uint64_t2 _e87 = input_uniform.val_u64_2_;
    uint64_t2 _e90 = input_storage.Load<uint64_t2>(32);
    output.Store(32, (_e87 + _e90));
    uint64_t3 _e96 = input_uniform.val_u64_3_;
    uint64_t3 _e99 = input_storage.Load<uint64_t3>(64);
    output.Store(64, (_e96 + _e99));
    uint64_t4 _e105 = input_uniform.val_u64_4_;
    uint64_t4 _e108 = input_storage.Load<uint64_t4>(96);
    output.Store(96, (_e105 + _e108));
    uint64_t _e114[2] = Constructarray2_uint64_t_(input_arrays.Load<uint64_t>(0+0), input_arrays.Load<uint64_t>(0+8));
    {
        uint64_t _value2[2] = _e114;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+8, _value2[1]);
    }
    uint64_t _e115 = val_1;
    uint64_t _e117 = val_1;
    val_1 = (_e117 + abs(_e115));
    uint64_t _e119 = val_1;
    uint64_t _e120 = val_1;
    uint64_t _e121 = val_1;
    uint64_t _e123 = val_1;
    val_1 = (_e123 + clamp(_e119, _e120, _e121));
    uint64_t _e125 = val_1;
    uint64_t _e127 = val_1;
    uint64_t _e130 = val_1;
    val_1 = (_e130 + dot((_e125).xx, (_e127).xx));
    uint64_t _e132 = val_1;
    uint64_t _e133 = val_1;
    uint64_t _e135 = val_1;
    val_1 = (_e135 + max(_e132, _e133));
    uint64_t _e137 = val_1;
    uint64_t _e138 = val_1;
    uint64_t _e140 = val_1;
    val_1 = (_e140 + min(_e137, _e138));
    uint64_t _e142 = val_1;
    return _e142;
}

[numthreads(1, 1, 1)]
void main()
{
    const uint64_t _e3 = uint64_function(67uL);
    const int64_t _e5 = int64_function(60L);
    output.Store(224, (_e3 + _e5));
    return;
}

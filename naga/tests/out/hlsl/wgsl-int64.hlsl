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

int64_t naga_f2i64(float value) {
    return int64_t(clamp(value, -9.223372e18, 9.2233715e18));
}

typedef int64_t ret_Constructarray2_int64_t_[2];
ret_Constructarray2_int64_t_ Constructarray2_int64_t_(int64_t arg0, int64_t arg1) {
    int64_t ret[2] = { arg0, arg1 };
    return ret;
}

int64_t int64_function(int64_t x)
{
    int64_t val = 20L;

    int64_t phony = private_variable;
    int64_t _e5 = val;
    val = (_e5 + ((31L - 1002003004005006L) + -9223372036854775807L));
    int64_t _e12 = val;
    int64_t _e13 = val;
    val = (_e12 + (_e13 + 5L));
    int64_t _e17 = val;
    uint _e20 = input_uniform.val_u32_;
    int64_t _e21 = val;
    val = (_e17 + int64_t((_e20 + uint(_e21))));
    int64_t _e26 = val;
    int _e29 = input_uniform.val_i32_;
    int64_t _e30 = val;
    val = (_e26 + int64_t(asint(asuint(_e29) + asuint(int(_e30)))));
    int64_t _e35 = val;
    float _e38 = input_uniform.val_f32_;
    int64_t _e39 = val;
    val = (_e35 + naga_f2i64((_e38 + float(_e39))));
    int64_t _e44 = val;
    int64_t _e47 = input_uniform.val_i64_;
    val = (_e44 + (_e47).xxx.z);
    int64_t _e51 = val;
    uint64_t _e54 = input_uniform.val_u64_;
    val = (_e51 + _e54);
    int64_t _e57 = val;
    uint64_t2 _e60 = input_uniform.val_u64_2_;
    val = (_e57 + _e60.y);
    int64_t _e64 = val;
    uint64_t3 _e67 = input_uniform.val_u64_3_;
    val = (_e64 + _e67.z);
    int64_t _e71 = val;
    uint64_t4 _e74 = input_uniform.val_u64_4_;
    val = (_e71 + _e74.w);
    int64_t _e78 = val;
    val = (_e78 + (-9223372036854775807L - 1L));
    int64_t _e85 = input_uniform.val_i64_;
    int64_t _e88 = input_storage.Load<int64_t>(128);
    output.Store(128, (_e85 + _e88));
    int64_t2 _e94 = input_uniform.val_i64_2_;
    int64_t2 _e97 = input_storage.Load<int64_t2>(144);
    output.Store(144, (_e94 + _e97));
    int64_t3 _e103 = input_uniform.val_i64_3_;
    int64_t3 _e106 = input_storage.Load<int64_t3>(160);
    output.Store(160, (_e103 + _e106));
    int64_t4 _e112 = input_uniform.val_i64_4_;
    int64_t4 _e115 = input_storage.Load<int64_t4>(192);
    output.Store(192, (_e112 + _e115));
    int64_t _e121[2] = Constructarray2_int64_t_(input_arrays.Load<int64_t>(16+0), input_arrays.Load<int64_t>(16+8));
    {
        int64_t _value2[2] = _e121;
        output_arrays.Store(16+0, _value2[0]);
        output_arrays.Store(16+8, _value2[1]);
    }
    int64_t _e122 = val;
    int64_t _e123 = val;
    val = (_e122 + abs(_e123));
    int64_t _e126 = val;
    int64_t _e127 = val;
    int64_t _e128 = val;
    int64_t _e129 = val;
    val = (_e126 + clamp(_e127, _e128, _e129));
    int64_t _e132 = val;
    int64_t _e133 = val;
    int64_t _e135 = val;
    val = (_e132 + dot((_e133).xx, (_e135).xx));
    int64_t _e139 = val;
    int64_t _e140 = val;
    int64_t _e141 = val;
    val = (_e139 + max(_e140, _e141));
    int64_t _e144 = val;
    int64_t _e145 = val;
    int64_t _e146 = val;
    val = (_e144 + min(_e145, _e146));
    int64_t _e149 = val;
    int64_t _e150 = val;
    val = (_e149 + sign(_e150));
    int64_t _e153 = val;
    return _e153;
}

uint64_t naga_f2u64(float value) {
    return uint64_t(clamp(value, 0.0, 1.8446743e19));
}

typedef uint64_t ret_Constructarray2_uint64_t_[2];
ret_Constructarray2_uint64_t_ Constructarray2_uint64_t_(uint64_t arg0, uint64_t arg1) {
    uint64_t ret[2] = { arg0, arg1 };
    return ret;
}

uint64_t uint64_function(uint64_t x_1)
{
    uint64_t val_1 = 20uL;

    uint64_t _e3 = val_1;
    val_1 = (_e3 + ((31uL + 18446744073709551615uL) - 18446744073709551615uL));
    uint64_t _e10 = val_1;
    uint64_t _e11 = val_1;
    val_1 = (_e10 + (_e11 + 5uL));
    uint64_t _e15 = val_1;
    uint _e18 = input_uniform.val_u32_;
    uint64_t _e19 = val_1;
    val_1 = (_e15 + uint64_t((_e18 + uint(_e19))));
    uint64_t _e24 = val_1;
    int _e27 = input_uniform.val_i32_;
    uint64_t _e28 = val_1;
    val_1 = (_e24 + uint64_t(asint(asuint(_e27) + asuint(int(_e28)))));
    uint64_t _e33 = val_1;
    float _e36 = input_uniform.val_f32_;
    uint64_t _e37 = val_1;
    val_1 = (_e33 + naga_f2u64((_e36 + float(_e37))));
    uint64_t _e42 = val_1;
    uint64_t _e45 = input_uniform.val_u64_;
    val_1 = (_e42 + (_e45).xxx.z);
    uint64_t _e49 = val_1;
    int64_t _e52 = input_uniform.val_i64_;
    val_1 = (_e49 + _e52);
    uint64_t _e55 = val_1;
    int64_t2 _e58 = input_uniform.val_i64_2_;
    val_1 = (_e55 + _e58.y);
    uint64_t _e62 = val_1;
    int64_t3 _e65 = input_uniform.val_i64_3_;
    val_1 = (_e62 + _e65.z);
    uint64_t _e69 = val_1;
    int64_t4 _e72 = input_uniform.val_i64_4_;
    val_1 = (_e69 + _e72.w);
    uint64_t _e80 = input_uniform.val_u64_;
    uint64_t _e83 = input_storage.Load<uint64_t>(16);
    output.Store(16, (_e80 + _e83));
    uint64_t2 _e89 = input_uniform.val_u64_2_;
    uint64_t2 _e92 = input_storage.Load<uint64_t2>(32);
    output.Store(32, (_e89 + _e92));
    uint64_t3 _e98 = input_uniform.val_u64_3_;
    uint64_t3 _e101 = input_storage.Load<uint64_t3>(64);
    output.Store(64, (_e98 + _e101));
    uint64_t4 _e107 = input_uniform.val_u64_4_;
    uint64_t4 _e110 = input_storage.Load<uint64_t4>(96);
    output.Store(96, (_e107 + _e110));
    uint64_t _e116[2] = Constructarray2_uint64_t_(input_arrays.Load<uint64_t>(0+0), input_arrays.Load<uint64_t>(0+8));
    {
        uint64_t _value2[2] = _e116;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+8, _value2[1]);
    }
    uint64_t _e117 = val_1;
    uint64_t _e118 = val_1;
    val_1 = (_e117 + abs(_e118));
    uint64_t _e121 = val_1;
    uint64_t _e122 = val_1;
    uint64_t _e123 = val_1;
    uint64_t _e124 = val_1;
    val_1 = (_e121 + clamp(_e122, _e123, _e124));
    uint64_t _e127 = val_1;
    uint64_t _e128 = val_1;
    uint64_t _e130 = val_1;
    val_1 = (_e127 + dot((_e128).xx, (_e130).xx));
    uint64_t _e134 = val_1;
    uint64_t _e135 = val_1;
    uint64_t _e136 = val_1;
    val_1 = (_e134 + max(_e135, _e136));
    uint64_t _e139 = val_1;
    uint64_t _e140 = val_1;
    uint64_t _e141 = val_1;
    val_1 = (_e139 + min(_e140, _e141));
    uint64_t _e144 = val_1;
    return _e144;
}

[numthreads(1, 1, 1)]
void main()
{
    const uint64_t _e3 = uint64_function(67uL);
    const int64_t _e5 = int64_function(60L);
    output.Store(224, (_e3 + _e5));
    return;
}

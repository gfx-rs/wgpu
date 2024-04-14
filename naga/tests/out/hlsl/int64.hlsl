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

    int64_t _expr6 = val;
    val = (_expr6 + (31L - 1002003004005006L));
    int64_t _expr8 = val;
    int64_t _expr11 = val;
    val = (_expr11 + (_expr8 + 5L));
    uint _expr15 = input_uniform.val_u32_;
    int64_t _expr16 = val;
    int64_t _expr20 = val;
    val = (_expr20 + int64_t((_expr15 + uint(_expr16))));
    int _expr24 = input_uniform.val_i32_;
    int64_t _expr25 = val;
    int64_t _expr29 = val;
    val = (_expr29 + int64_t((_expr24 + int(_expr25))));
    float _expr33 = input_uniform.val_f32_;
    int64_t _expr34 = val;
    int64_t _expr38 = val;
    val = (_expr38 + int64_t((_expr33 + float(_expr34))));
    int64_t _expr42 = input_uniform.val_i64_;
    int64_t _expr45 = val;
    val = (_expr45 + (_expr42).xxx.z);
    uint64_t _expr49 = input_uniform.val_u64_;
    int64_t _expr51 = val;
    val = (_expr51 + _expr49);
    uint64_t2 _expr55 = input_uniform.val_u64_2_;
    int64_t _expr58 = val;
    val = (_expr58 + _expr55.y);
    uint64_t3 _expr62 = input_uniform.val_u64_3_;
    int64_t _expr65 = val;
    val = (_expr65 + _expr62.z);
    uint64_t4 _expr69 = input_uniform.val_u64_4_;
    int64_t _expr72 = val;
    val = (_expr72 + _expr69.w);
    int64_t _expr78 = input_uniform.val_i64_;
    int64_t _expr81 = input_storage.Load<int64_t>(128);
    output.Store(128, (_expr78 + _expr81));
    int64_t2 _expr87 = input_uniform.val_i64_2_;
    int64_t2 _expr90 = input_storage.Load<int64_t2>(144);
    output.Store(144, (_expr87 + _expr90));
    int64_t3 _expr96 = input_uniform.val_i64_3_;
    int64_t3 _expr99 = input_storage.Load<int64_t3>(160);
    output.Store(160, (_expr96 + _expr99));
    int64_t4 _expr105 = input_uniform.val_i64_4_;
    int64_t4 _expr108 = input_storage.Load<int64_t4>(192);
    output.Store(192, (_expr105 + _expr108));
    int64_t _expr114[2] = Constructarray2_int64_t_(input_arrays.Load<int64_t>(16+0), input_arrays.Load<int64_t>(16+8));
    {
        int64_t _value2[2] = _expr114;
        output_arrays.Store(16+0, _value2[0]);
        output_arrays.Store(16+8, _value2[1]);
    }
    int64_t _expr115 = val;
    int64_t _expr117 = val;
    val = (_expr117 + abs(_expr115));
    int64_t _expr119 = val;
    int64_t _expr120 = val;
    int64_t _expr121 = val;
    int64_t _expr123 = val;
    val = (_expr123 + clamp(_expr119, _expr120, _expr121));
    int64_t _expr125 = val;
    int64_t _expr127 = val;
    int64_t _expr130 = val;
    val = (_expr130 + dot((_expr125).xx, (_expr127).xx));
    int64_t _expr132 = val;
    int64_t _expr133 = val;
    int64_t _expr135 = val;
    val = (_expr135 + max(_expr132, _expr133));
    int64_t _expr137 = val;
    int64_t _expr138 = val;
    int64_t _expr140 = val;
    val = (_expr140 + min(_expr137, _expr138));
    int64_t _expr142 = val;
    int64_t _expr144 = val;
    val = (_expr144 + sign(_expr142));
    int64_t _expr146 = val;
    return _expr146;
}

typedef uint64_t ret_Constructarray2_uint64_t_[2];
ret_Constructarray2_uint64_t_ Constructarray2_uint64_t_(uint64_t arg0, uint64_t arg1) {
    uint64_t ret[2] = { arg0, arg1 };
    return ret;
}

uint64_t uint64_function(uint64_t x_1)
{
    uint64_t val_1 = 20uL;

    uint64_t _expr6 = val_1;
    val_1 = (_expr6 + (31uL + 1002003004005006uL));
    uint64_t _expr8 = val_1;
    uint64_t _expr11 = val_1;
    val_1 = (_expr11 + (_expr8 + 5uL));
    uint _expr15 = input_uniform.val_u32_;
    uint64_t _expr16 = val_1;
    uint64_t _expr20 = val_1;
    val_1 = (_expr20 + uint64_t((_expr15 + uint(_expr16))));
    int _expr24 = input_uniform.val_i32_;
    uint64_t _expr25 = val_1;
    uint64_t _expr29 = val_1;
    val_1 = (_expr29 + uint64_t((_expr24 + int(_expr25))));
    float _expr33 = input_uniform.val_f32_;
    uint64_t _expr34 = val_1;
    uint64_t _expr38 = val_1;
    val_1 = (_expr38 + uint64_t((_expr33 + float(_expr34))));
    uint64_t _expr42 = input_uniform.val_u64_;
    uint64_t _expr45 = val_1;
    val_1 = (_expr45 + (_expr42).xxx.z);
    int64_t _expr49 = input_uniform.val_i64_;
    uint64_t _expr51 = val_1;
    val_1 = (_expr51 + _expr49);
    int64_t2 _expr55 = input_uniform.val_i64_2_;
    uint64_t _expr58 = val_1;
    val_1 = (_expr58 + _expr55.y);
    int64_t3 _expr62 = input_uniform.val_i64_3_;
    uint64_t _expr65 = val_1;
    val_1 = (_expr65 + _expr62.z);
    int64_t4 _expr69 = input_uniform.val_i64_4_;
    uint64_t _expr72 = val_1;
    val_1 = (_expr72 + _expr69.w);
    uint64_t _expr78 = input_uniform.val_u64_;
    uint64_t _expr81 = input_storage.Load<uint64_t>(16);
    output.Store(16, (_expr78 + _expr81));
    uint64_t2 _expr87 = input_uniform.val_u64_2_;
    uint64_t2 _expr90 = input_storage.Load<uint64_t2>(32);
    output.Store(32, (_expr87 + _expr90));
    uint64_t3 _expr96 = input_uniform.val_u64_3_;
    uint64_t3 _expr99 = input_storage.Load<uint64_t3>(64);
    output.Store(64, (_expr96 + _expr99));
    uint64_t4 _expr105 = input_uniform.val_u64_4_;
    uint64_t4 _expr108 = input_storage.Load<uint64_t4>(96);
    output.Store(96, (_expr105 + _expr108));
    uint64_t _expr114[2] = Constructarray2_uint64_t_(input_arrays.Load<uint64_t>(0+0), input_arrays.Load<uint64_t>(0+8));
    {
        uint64_t _value2[2] = _expr114;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+8, _value2[1]);
    }
    uint64_t _expr115 = val_1;
    uint64_t _expr117 = val_1;
    val_1 = (_expr117 + abs(_expr115));
    uint64_t _expr119 = val_1;
    uint64_t _expr120 = val_1;
    uint64_t _expr121 = val_1;
    uint64_t _expr123 = val_1;
    val_1 = (_expr123 + clamp(_expr119, _expr120, _expr121));
    uint64_t _expr125 = val_1;
    uint64_t _expr127 = val_1;
    uint64_t _expr130 = val_1;
    val_1 = (_expr130 + dot((_expr125).xx, (_expr127).xx));
    uint64_t _expr132 = val_1;
    uint64_t _expr133 = val_1;
    uint64_t _expr135 = val_1;
    val_1 = (_expr135 + max(_expr132, _expr133));
    uint64_t _expr137 = val_1;
    uint64_t _expr138 = val_1;
    uint64_t _expr140 = val_1;
    val_1 = (_expr140 + min(_expr137, _expr138));
    uint64_t _expr142 = val_1;
    return _expr142;
}

[numthreads(1, 1, 1)]
void main()
{
    const uint64_t _e3 = uint64_function(67uL);
    const int64_t _e5 = int64_function(60L);
    output.Store(224, (_e3 + _e5));
    return;
}

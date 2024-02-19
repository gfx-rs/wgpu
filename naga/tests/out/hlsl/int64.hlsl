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
    double val_f64_;
    uint64_t val_u64_;
    uint64_t2 val_u64_2_;
    int _pad6_0;
    int _pad6_1;
    int _pad6_2;
    int _pad6_3;
    uint64_t3 val_u64_3_;
    int _pad7_0;
    int _pad7_1;
    uint64_t4 val_u64_4_;
    int64_t val_i64_;
    int _pad9_0;
    int _pad9_1;
    int64_t2 val_i64_2_;
    int64_t3 val_i64_3_;
    int _pad11_0;
    int _pad11_1;
    int64_t4 val_i64_4_;
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
    int64_t val_1_ = (31L - 1002003004005006L);
    int64_t val_2_ = (val_1_ + 5L);
    uint _expr8 = input_uniform.val_u32_;
    int _expr12 = input_uniform.val_i32_;
    float _expr17 = input_uniform.val_f32_;
    double _expr22 = input_uniform.val_f64_;
    int64_t val_3_ = (((int64_t(_expr8) + int64_t(_expr12)) + int64_t(_expr17)) + int64_t(_expr22));
    int64_t _expr27 = input_uniform.val_i64_;
    int64_t3 val_4_ = (_expr27).xxx;
    int64_t _expr33 = input_uniform.val_i64_;
    int64_t _expr36 = input_storage.Load<int64_t>(128);
    output.Store(128, (_expr33 + _expr36));
    int64_t2 _expr42 = input_uniform.val_i64_2_;
    int64_t2 _expr45 = input_storage.Load<int64_t2>(144);
    output.Store(144, (_expr42 + _expr45));
    int64_t3 _expr51 = input_uniform.val_i64_3_;
    int64_t3 _expr54 = input_storage.Load<int64_t3>(160);
    output.Store(160, (_expr51 + _expr54));
    int64_t4 _expr60 = input_uniform.val_i64_4_;
    int64_t4 _expr63 = input_storage.Load<int64_t4>(192);
    output.Store(192, (_expr60 + _expr63));
    int64_t _expr69[2] = Constructarray2_int64_t_(input_arrays.Load<int64_t>(16+0), input_arrays.Load<int64_t>(16+8));
    {
        int64_t _value2[2] = _expr69;
        output_arrays.Store(16+0, _value2[0]);
        output_arrays.Store(16+8, _value2[1]);
    }
    return ((((val_1_ + val_2_) + val_3_) + 20L) + 50L);
}

typedef uint64_t ret_Constructarray2_uint64_t_[2];
ret_Constructarray2_uint64_t_ Constructarray2_uint64_t_(uint64_t arg0, uint64_t arg1) {
    uint64_t ret[2] = { arg0, arg1 };
    return ret;
}

uint64_t uint64_function(uint64_t x_1)
{
    uint64_t val_1_1 = (31uL + 1002003004005006uL);
    uint64_t val_2_1 = (val_1_1 + 5uL);
    uint _expr8 = input_uniform.val_u32_;
    int _expr12 = input_uniform.val_i32_;
    float _expr17 = input_uniform.val_f32_;
    double _expr22 = input_uniform.val_f64_;
    uint64_t val_3_1 = (((uint64_t(_expr8) + uint64_t(_expr12)) + uint64_t(_expr17)) + uint64_t(_expr22));
    uint64_t _expr27 = input_uniform.val_u64_;
    uint64_t3 val_4_1 = (_expr27).xxx;
    uint64_t _expr33 = input_uniform.val_u64_;
    uint64_t _expr36 = input_storage.Load<uint64_t>(24);
    output.Store(24, (_expr33 + _expr36));
    uint64_t2 _expr42 = input_uniform.val_u64_2_;
    uint64_t2 _expr45 = input_storage.Load<uint64_t2>(32);
    output.Store(32, (_expr42 + _expr45));
    uint64_t3 _expr51 = input_uniform.val_u64_3_;
    uint64_t3 _expr54 = input_storage.Load<uint64_t3>(64);
    output.Store(64, (_expr51 + _expr54));
    uint64_t4 _expr60 = input_uniform.val_u64_4_;
    uint64_t4 _expr63 = input_storage.Load<uint64_t4>(96);
    output.Store(96, (_expr60 + _expr63));
    uint64_t _expr69[2] = Constructarray2_uint64_t_(input_arrays.Load<uint64_t>(0+0), input_arrays.Load<uint64_t>(0+8));
    {
        uint64_t _value2[2] = _expr69;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+8, _value2[1]);
    }
    return ((((val_1_1 + val_2_1) + val_3_1) + 20uL) + 50uL);
}

[numthreads(1, 1, 1)]
void main()
{
    const uint64_t _e1 = uint64_function(67uL);
    const int64_t _e3 = int64_function(60L);
    return;
}

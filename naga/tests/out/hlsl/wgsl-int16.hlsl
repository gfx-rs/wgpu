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
    uint16_t val_u16_;
    uint16_t2 val_u16_2_;
    int _pad5_0;
    uint16_t3 val_u16_3_;
    uint16_t4 val_u16_4_;
    int16_t val_i16_;
    int16_t2 val_i16_2_;
    int16_t3 val_i16_3_;
    int16_t4 val_i16_4_;
    uint16_t final_value;
    int _end_pad_0;
};

struct StorageCompatible {
    uint16_t val_u16_array_2_[2];
    int16_t val_i16_array_2_[2];
};

static const uint16_t constant_variable = uint16_t(20);
static const int16_t f16_to_i16_clamped = int16_t(32767);

static int16_t private_variable = int16_t(1);
cbuffer input_uniform : register(b0) { UniformCompatible input_uniform; }
ByteAddressBuffer input_storage : register(t1);
ByteAddressBuffer input_arrays : register(t2);
RWByteAddressBuffer output : register(u3);
RWByteAddressBuffer output_arrays : register(u4);
groupshared uint16_t shared_val;

struct ComputeInput_main {
};

int16_t naga_div(int16_t lhs, int16_t rhs) {
    return lhs / (((lhs == int16_t(-32768) & rhs == -1) | (rhs == 0)) ? 1 : rhs);
}

int16_t naga_mod(int16_t lhs, int16_t rhs) {
    int16_t divisor = ((lhs == int16_t(-32768) & rhs == -1) | (rhs == 0)) ? 1 : rhs;
    return lhs - (lhs / divisor) * divisor;
}

typedef int16_t ret_Constructarray4_int16_t_[4];
ret_Constructarray4_int16_t_ Constructarray4_int16_t_(int16_t arg0, int16_t arg1, int16_t arg2, int16_t arg3) {
    int16_t ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef int16_t ret_Constructarray2_int16_t_[2];
ret_Constructarray2_int16_t_ Constructarray2_int16_t_(int16_t arg0, int16_t arg1) {
    int16_t ret[2] = { arg0, arg1 };
    return ret;
}

int16_t int16_function(int16_t x)
{
    int16_t val = int16_t(20);
    int16_t arr[4] = Constructarray4_int16_t_(int16_t(1), int16_t(2), int16_t(3), int16_t(4));

    int16_t phony = private_variable;
    int16_t _e5 = val;
    val = (_e5 + int16_t(5));
    int16_t _e8 = val;
    uint _e11 = input_uniform.val_u32_;
    val = (_e8 + int16_t(_e11));
    int16_t _e14 = val;
    int _e17 = input_uniform.val_i32_;
    val = (_e14 + int16_t(_e17));
    int16_t _e20 = val;
    int16_t _e23 = input_uniform.val_i16_;
    val = (_e20 + (_e23).xxx.z);
    int16_t _e31 = input_uniform.val_i16_;
    int16_t _e34 = input_storage.Load<int16_t>(40);
    output.Store(40, (_e31 + _e34));
    int16_t2 _e40 = input_uniform.val_i16_2_;
    int16_t2 _e43 = input_storage.Load<int16_t2>(44);
    output.Store(44, (_e40 + _e43));
    int16_t3 _e49 = input_uniform.val_i16_3_;
    int16_t3 _e52 = input_storage.Load<int16_t3>(48);
    output.Store(48, (_e49 + _e52));
    int16_t4 _e58 = input_uniform.val_i16_4_;
    int16_t4 _e61 = input_storage.Load<int16_t4>(56);
    output.Store(56, (_e58 + _e61));
    int16_t _e67[2] = Constructarray2_int16_t_(input_arrays.Load<int16_t>(4+0), input_arrays.Load<int16_t>(4+2));
    {
        int16_t _value2[2] = _e67;
        output_arrays.Store(4+0, _value2[0]);
        output_arrays.Store(4+2, _value2[1]);
    }
    int16_t _e68 = val;
    val = abs(_e68);
    int16_t _e70 = val;
    int16_t _e71 = val;
    val = max(_e70, _e71);
    int16_t _e73 = val;
    int16_t _e74 = val;
    val = min(_e73, _e74);
    int16_t _e76 = val;
    int16_t _e77 = val;
    int16_t _e78 = val;
    val = clamp(_e76, _e77, _e78);
    int16_t _e80 = val;
    val = sign(_e80);
    int16_t _e82 = val;
    val = (_e82 - int16_t(1));
    int16_t _e85 = val;
    val = (_e85 * int16_t(2));
    int16_t _e88 = val;
    val = naga_div(_e88, int16_t(3));
    int16_t _e91 = val;
    val = naga_mod(_e91, int16_t(4));
    int16_t _e94 = val;
    val = (_e94 & int16_t(255));
    int16_t _e97 = val;
    val = (_e97 | int16_t(16));
    int16_t _e100 = val;
    val = (_e100 ^ int16_t(1));
    int16_t _e103 = val;
    val = (_e103 << 2u);
    int16_t _e106 = val;
    val = (_e106 >> 1u);
    int16_t _e109 = val;
    val = -(_e109);
    int16_t _e111 = val;
    bool cmp_lt = (_e111 < int16_t(0));
    int16_t _e114 = val;
    bool cmp_le = (_e114 <= int16_t(0));
    int16_t _e117 = val;
    bool cmp_gt = (_e117 > int16_t(0));
    int16_t _e120 = val;
    bool cmp_ge = (_e120 >= int16_t(0));
    int16_t _e123 = val;
    bool cmp_eq = (_e123 == int16_t(0));
    int16_t _e126 = val;
    bool cmp_ne = (_e126 != int16_t(0));
    val = (cmp_lt ? int16_t(2) : int16_t(1));
    int16_t _e139 = val;
    arr[0] = _e139;
    int16_t _e141 = arr[1];
    val = _e141;
    int16_t _e144 = arr[uint16_t(1)];
    val = _e144;
    int16_t _e147 = val;
    output.Store(0, asuint(uint(_e147)));
    int16_t _e151 = val;
    output.Store(4, asuint(int(_e151)));
    int16_t _e155 = val;
    output.Store(8, asuint(float(_e155)));
    uint _e159 = asuint(output.Load(0));
    val = int16_t(_e159);
    int16_t _e161 = val;
    uint16_t as_unsigned = uint16_t(_e161);
    val = int16_t(as_unsigned);
    int16_t2 _e166 = input_uniform.val_i16_2_;
    int16_t2 _e169 = input_uniform.val_i16_2_;
    int16_t2 v = (_e166 + _e169);
    int16_t2 v2_ = (v * (int16_t(2)).xx);
    output.Store(44, v2_);
    int16_t _e176 = val;
    return _e176;
}

uint16_t naga_div(uint16_t lhs, uint16_t rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

uint16_t naga_mod(uint16_t lhs, uint16_t rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

typedef uint16_t ret_Constructarray2_uint16_t_[2];
ret_Constructarray2_uint16_t_ Constructarray2_uint16_t_(uint16_t arg0, uint16_t arg1) {
    uint16_t ret[2] = { arg0, arg1 };
    return ret;
}

uint16_t uint16_function(uint16_t x_1)
{
    uint16_t val_1 = uint16_t(20);

    uint16_t _e3 = val_1;
    val_1 = (_e3 + uint16_t(5));
    uint16_t _e6 = val_1;
    uint _e9 = input_uniform.val_u32_;
    val_1 = (_e6 + uint16_t(_e9));
    uint16_t _e12 = val_1;
    int _e15 = input_uniform.val_i32_;
    val_1 = (_e12 + uint16_t(_e15));
    uint16_t _e18 = val_1;
    uint16_t _e21 = input_uniform.val_u16_;
    val_1 = (_e18 + (_e21).xxx.z);
    uint16_t _e29 = input_uniform.val_u16_;
    uint16_t _e32 = input_storage.Load<uint16_t>(12);
    output.Store(12, (_e29 + _e32));
    uint16_t2 _e38 = input_uniform.val_u16_2_;
    uint16_t2 _e41 = input_storage.Load<uint16_t2>(16);
    output.Store(16, (_e38 + _e41));
    uint16_t3 _e47 = input_uniform.val_u16_3_;
    uint16_t3 _e50 = input_storage.Load<uint16_t3>(24);
    output.Store(24, (_e47 + _e50));
    uint16_t4 _e56 = input_uniform.val_u16_4_;
    uint16_t4 _e59 = input_storage.Load<uint16_t4>(32);
    output.Store(32, (_e56 + _e59));
    uint16_t _e65[2] = Constructarray2_uint16_t_(input_arrays.Load<uint16_t>(0+0), input_arrays.Load<uint16_t>(0+2));
    {
        uint16_t _value2[2] = _e65;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    uint16_t _e66 = val_1;
    val_1 = abs(_e66);
    uint16_t _e68 = val_1;
    uint16_t _e69 = val_1;
    val_1 = max(_e68, _e69);
    uint16_t _e71 = val_1;
    uint16_t _e72 = val_1;
    val_1 = min(_e71, _e72);
    uint16_t _e74 = val_1;
    uint16_t _e75 = val_1;
    uint16_t _e76 = val_1;
    val_1 = clamp(_e74, _e75, _e76);
    uint16_t _e78 = val_1;
    val_1 = (_e78 - uint16_t(1));
    uint16_t _e81 = val_1;
    val_1 = (_e81 * uint16_t(2));
    uint16_t _e84 = val_1;
    val_1 = naga_div(_e84, uint16_t(3));
    uint16_t _e87 = val_1;
    val_1 = naga_mod(_e87, uint16_t(4));
    uint16_t _e90 = val_1;
    val_1 = (_e90 & uint16_t(255));
    uint16_t _e93 = val_1;
    val_1 = (_e93 | uint16_t(16));
    uint16_t _e96 = val_1;
    val_1 = (_e96 ^ uint16_t(1));
    uint16_t _e101 = val_1;
    output.Store(0, asuint(uint(_e101)));
    uint16_t _e105 = val_1;
    output.Store(4, asuint(int(_e105)));
    uint16_t _e109 = val_1;
    output.Store(8, asuint(float(_e109)));
    uint _e113 = asuint(output.Load(0));
    val_1 = uint16_t(_e113);
    uint16_t _e115 = val_1;
    return _e115;
}

[numthreads(64, 1, 1)]
void main(ComputeInput_main computeinput_main, uint local_invocation_index : SV_GroupIndex)
{
    if (local_invocation_index == 0) {
        shared_val = (uint16_t)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint subgroup_invocation_id = WaveGetLaneIndex();
    int16_t sg_val = (int16_t)0;
    uint16_t sg_uval = (uint16_t)0;

    shared_val = uint16_t(0);
    const uint16_t _e6 = uint16_function(uint16_t(67));
    const int16_t _e8 = int16_function(int16_t(60));
    output.Store(64, (_e6 + uint16_t(_e8)));
    sg_val = int16_t(subgroup_invocation_id);
    int16_t _e13 = sg_val;
    const int16_t _e14 = WaveActiveSum(_e13);
    sg_val = _e14;
    int16_t _e15 = sg_val;
    const int16_t _e16 = WaveActiveProduct(_e15);
    sg_val = _e16;
    int16_t _e17 = sg_val;
    const int16_t _e18 = WaveActiveMin(_e17);
    sg_val = _e18;
    int16_t _e19 = sg_val;
    const int16_t _e20 = WaveActiveMax(_e19);
    sg_val = _e20;
    int16_t _e21 = sg_val;
    const int16_t _e22 = WavePrefixSum(_e21);
    sg_val = _e22;
    int16_t _e23 = sg_val;
    const int16_t _e24 = _e23 + WavePrefixSum(_e23);
    sg_val = _e24;
    int16_t _e25 = sg_val;
    const int16_t _e26 = WaveReadLaneFirst(_e25);
    sg_val = _e26;
    int16_t _e27 = sg_val;
    const int16_t _e29 = WaveReadLaneAt(_e27, 4u);
    sg_val = _e29;
    sg_uval = uint16_t(subgroup_invocation_id);
    uint16_t _e32 = sg_uval;
    const uint16_t _e33 = WaveActiveSum(_e32);
    sg_uval = _e33;
    uint16_t _e34 = sg_uval;
    const uint16_t _e35 = WaveActiveMin(_e34);
    sg_uval = _e35;
    uint16_t _e36 = sg_uval;
    const uint16_t _e37 = WaveActiveMax(_e36);
    sg_uval = _e37;
    int16_t _e40 = sg_val;
    output.Store(40, _e40);
    uint16_t _e43 = sg_uval;
    output.Store(12, _e43);
    return;
}

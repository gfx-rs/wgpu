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
    uint16_t _pad4_0;
    uint16_t2 val_u16_2_;
    int _pad5_0;
    uint16_t3 val_u16_3_;
    uint16_t _pad6_0;
    uint16_t4 val_u16_4_;
    int16_t val_i16_;
    uint16_t _pad8_0;
    int16_t2 val_i16_2_;
    int16_t3 val_i16_3_;
    uint16_t _pad10_0;
    int16_t4 val_i16_4_;
    uint16_t final_value;
    uint16_t _end_pad_0;
    int _end_pad_1;
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

uint16_t naga_div(uint16_t lhs, uint16_t rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

uint16_t naga_mod(uint16_t lhs, uint16_t rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

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

typedef uint16_t ret_Constructarray2_uint16_t_[2];
ret_Constructarray2_uint16_t_ Constructarray2_uint16_t_(uint16_t arg0, uint16_t arg1) {
    uint16_t ret[2] = { arg0, arg1 };
    return ret;
}

typedef int16_t ret_Constructarray2_int16_t_[2];
ret_Constructarray2_int16_t_ Constructarray2_int16_t_(int16_t arg0, int16_t arg1) {
    int16_t ret[2] = { arg0, arg1 };
    return ret;
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
    uint16_t x = (uint16_t)0;
    uint16_t val = (uint16_t)0;
    uint16_t local = (uint16_t)0;
    int16_t x_1 = (int16_t)0;
    int16_t val_1 = (int16_t)0;
    int16_t arr[4] = (int16_t[4])0;
    int16_t local_1 = (int16_t)0;

    shared_val = uint16_t(0);
    x = uint16_t(67);
    val = uint16_t(20);
    uint16_t _e3 = val;
    val = (_e3 + uint16_t(5));
    uint16_t _e6 = val;
    uint _e9 = input_uniform.val_u32_;
    val = (_e6 + uint16_t(_e9));
    uint16_t _e12 = val;
    int _e15 = input_uniform.val_i32_;
    val = (_e12 + uint16_t(_e15));
    uint16_t _e18 = val;
    uint16_t _e21 = input_uniform.val_u16_;
    val = (_e18 + (_e21).xxx.z);
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
    uint16_t _e66 = val;
    val = abs(_e66);
    uint16_t _e68 = val;
    uint16_t _e69 = val;
    val = max(_e68, _e69);
    uint16_t _e71 = val;
    uint16_t _e72 = val;
    val = min(_e71, _e72);
    uint16_t _e74 = val;
    uint16_t _e75 = val;
    uint16_t _e76 = val;
    val = clamp(_e74, _e75, _e76);
    uint16_t _e78 = val;
    val = (_e78 - uint16_t(1));
    uint16_t _e81 = val;
    val = (_e81 * uint16_t(2));
    uint16_t _e84 = val;
    val = naga_div(_e84, uint16_t(3));
    uint16_t _e87 = val;
    val = naga_mod(_e87, uint16_t(4));
    uint16_t _e90 = val;
    val = (_e90 & uint16_t(255));
    uint16_t _e93 = val;
    val = (_e93 | uint16_t(16));
    uint16_t _e96 = val;
    val = (_e96 ^ uint16_t(1));
    uint16_t _e101 = val;
    output.Store(0, asuint(uint(_e101)));
    uint16_t _e105 = val;
    output.Store(4, asuint(int(_e105)));
    uint16_t _e109 = val;
    output.Store(8, asuint(float(_e109)));
    uint _e113 = asuint(output.Load(0));
    val = uint16_t(_e113);
    uint16_t _e115 = val;
    local = _e115;
    uint16_t _e117 = local;
    x_1 = int16_t(60);
    val_1 = int16_t(20);
    arr = Constructarray4_int16_t_(int16_t(1), int16_t(2), int16_t(3), int16_t(4));
    int16_t _e124 = val_1;
    val_1 = (_e124 + int16_t(5));
    int16_t _e127 = val_1;
    uint _e130 = input_uniform.val_u32_;
    val_1 = (_e127 + int16_t(_e130));
    int16_t _e133 = val_1;
    int _e136 = input_uniform.val_i32_;
    val_1 = (_e133 + int16_t(_e136));
    int16_t _e139 = val_1;
    int16_t _e142 = input_uniform.val_i16_;
    val_1 = (_e139 + (_e142).xxx.z);
    int16_t _e150 = input_uniform.val_i16_;
    int16_t _e153 = input_storage.Load<int16_t>(40);
    output.Store(40, (_e150 + _e153));
    int16_t2 _e159 = input_uniform.val_i16_2_;
    int16_t2 _e162 = input_storage.Load<int16_t2>(44);
    output.Store(44, (_e159 + _e162));
    int16_t3 _e168 = input_uniform.val_i16_3_;
    int16_t3 _e171 = input_storage.Load<int16_t3>(48);
    output.Store(48, (_e168 + _e171));
    int16_t4 _e177 = input_uniform.val_i16_4_;
    int16_t4 _e180 = input_storage.Load<int16_t4>(56);
    output.Store(56, (_e177 + _e180));
    int16_t _e186[2] = Constructarray2_int16_t_(input_arrays.Load<int16_t>(4+0), input_arrays.Load<int16_t>(4+2));
    {
        int16_t _value2[2] = _e186;
        output_arrays.Store(4+0, _value2[0]);
        output_arrays.Store(4+2, _value2[1]);
    }
    int16_t _e187 = val_1;
    val_1 = abs(_e187);
    int16_t _e189 = val_1;
    int16_t _e190 = val_1;
    val_1 = max(_e189, _e190);
    int16_t _e192 = val_1;
    int16_t _e193 = val_1;
    val_1 = min(_e192, _e193);
    int16_t _e195 = val_1;
    int16_t _e196 = val_1;
    int16_t _e197 = val_1;
    val_1 = clamp(_e195, _e196, _e197);
    int16_t _e199 = val_1;
    val_1 = sign(_e199);
    int16_t _e201 = val_1;
    val_1 = (_e201 - int16_t(1));
    int16_t _e204 = val_1;
    val_1 = (_e204 * int16_t(2));
    int16_t _e207 = val_1;
    val_1 = naga_div(_e207, int16_t(3));
    int16_t _e210 = val_1;
    val_1 = naga_mod(_e210, int16_t(4));
    int16_t _e213 = val_1;
    val_1 = (_e213 & int16_t(255));
    int16_t _e216 = val_1;
    val_1 = (_e216 | int16_t(16));
    int16_t _e219 = val_1;
    val_1 = (_e219 ^ int16_t(1));
    int16_t _e222 = val_1;
    val_1 = (_e222 << 2u);
    int16_t _e225 = val_1;
    val_1 = (_e225 >> 1u);
    int16_t _e228 = val_1;
    val_1 = -(_e228);
    int16_t _e230 = val_1;
    val_1 = ((_e230 < int16_t(0)) ? int16_t(2) : int16_t(1));
    int16_t _e238 = val_1;
    arr[0] = _e238;
    int16_t _e240 = arr[1];
    val_1 = _e240;
    int16_t _e243 = arr[uint16_t(1)];
    val_1 = _e243;
    int16_t _e246 = val_1;
    output.Store(0, asuint(uint(_e246)));
    int16_t _e250 = val_1;
    output.Store(4, asuint(int(_e250)));
    int16_t _e254 = val_1;
    output.Store(8, asuint(float(_e254)));
    uint _e258 = asuint(output.Load(0));
    val_1 = int16_t(_e258);
    int16_t _e260 = val_1;
    val_1 = int16_t(uint16_t(_e260));
    int16_t2 _e265 = input_uniform.val_i16_2_;
    int16_t2 _e268 = input_uniform.val_i16_2_;
    output.Store(44, ((_e265 + _e268) * (int16_t(2)).xx));
    int16_t _e275 = val_1;
    local_1 = _e275;
    int16_t _e277 = local_1;
    output.Store(64, (_e117 + uint16_t(_e277)));
    sg_val = int16_t(subgroup_invocation_id);
    int16_t _e283 = sg_val;
    const int16_t _e306 = WaveActiveSum(_e283);
    sg_val = _e306;
    int16_t _e284 = sg_val;
    const int16_t _e307 = WaveActiveProduct(_e284);
    sg_val = _e307;
    int16_t _e285 = sg_val;
    const int16_t _e308 = WaveActiveMin(_e285);
    sg_val = _e308;
    int16_t _e286 = sg_val;
    const int16_t _e309 = WaveActiveMax(_e286);
    sg_val = _e309;
    int16_t _e287 = sg_val;
    const int16_t _e310 = WavePrefixSum(_e287);
    sg_val = _e310;
    int16_t _e288 = sg_val;
    const int16_t _e311 = _e288 + WavePrefixSum(_e288);
    sg_val = _e311;
    int16_t _e289 = sg_val;
    const int16_t _e312 = WaveReadLaneFirst(_e289);
    sg_val = _e312;
    int16_t _e290 = sg_val;
    const int16_t _e314 = WaveReadLaneAt(_e290, 4u);
    sg_val = _e314;
    sg_uval = uint16_t(subgroup_invocation_id);
    uint16_t _e293 = sg_uval;
    const uint16_t _e315 = WaveActiveSum(_e293);
    sg_uval = _e315;
    uint16_t _e294 = sg_uval;
    const uint16_t _e316 = WaveActiveMin(_e294);
    sg_uval = _e316;
    uint16_t _e295 = sg_uval;
    const uint16_t _e317 = WaveActiveMax(_e295);
    sg_uval = _e317;
    int16_t _e298 = sg_val;
    output.Store(40, _e298);
    uint16_t _e301 = sg_uval;
    output.Store(12, _e301);
    return;
}

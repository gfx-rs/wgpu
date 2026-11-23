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
    uint local_invocation_index_1 : SV_GroupIndex;
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
void main(ComputeInput_main computeinput_main)
{
    uint subgroup_invocation_id = WaveGetLaneIndex();
    uint local_invocation_index = computeinput_main.local_invocation_index_1;
    int16_t sg_val = (int16_t)0;
    uint16_t sg_uval = (uint16_t)0;
    uint16_t x = (uint16_t)0;
    uint16_t val = (uint16_t)0;
    uint16_t local = (uint16_t)0;
    int16_t x_1 = (int16_t)0;
    int16_t val_1 = (int16_t)0;
    int16_t arr[4] = (int16_t[4])0;
    int16_t local_1 = (int16_t)0;

    if ((local_invocation_index == 0u)) {
        shared_val = uint16_t(0);
    }
    GroupMemoryBarrierWithGroupSync();
    shared_val = uint16_t(0);
    x = uint16_t(67);
    val = uint16_t(20);
    uint16_t _e6 = val;
    val = (_e6 + uint16_t(5));
    uint16_t _e9 = val;
    uint _e12 = input_uniform.val_u32_;
    val = (_e9 + uint16_t(_e12));
    uint16_t _e15 = val;
    int _e18 = input_uniform.val_i32_;
    val = (_e15 + uint16_t(_e18));
    uint16_t _e21 = val;
    uint16_t _e24 = input_uniform.val_u16_;
    val = (_e21 + (_e24).xxx.z);
    uint16_t _e32 = input_uniform.val_u16_;
    uint16_t _e35 = input_storage.Load<uint16_t>(12);
    output.Store(12, (_e32 + _e35));
    uint16_t2 _e41 = input_uniform.val_u16_2_;
    uint16_t2 _e44 = input_storage.Load<uint16_t2>(16);
    output.Store(16, (_e41 + _e44));
    uint16_t3 _e50 = input_uniform.val_u16_3_;
    uint16_t3 _e53 = input_storage.Load<uint16_t3>(24);
    output.Store(24, (_e50 + _e53));
    uint16_t4 _e59 = input_uniform.val_u16_4_;
    uint16_t4 _e62 = input_storage.Load<uint16_t4>(32);
    output.Store(32, (_e59 + _e62));
    uint16_t _e68[2] = Constructarray2_uint16_t_(input_arrays.Load<uint16_t>(0+0), input_arrays.Load<uint16_t>(0+2));
    {
        uint16_t _value2[2] = _e68;
        output_arrays.Store(0+0, _value2[0]);
        output_arrays.Store(0+2, _value2[1]);
    }
    uint16_t _e69 = val;
    val = abs(_e69);
    uint16_t _e71 = val;
    uint16_t _e72 = val;
    val = max(_e71, _e72);
    uint16_t _e74 = val;
    uint16_t _e75 = val;
    val = min(_e74, _e75);
    uint16_t _e77 = val;
    uint16_t _e78 = val;
    uint16_t _e79 = val;
    val = clamp(_e77, _e78, _e79);
    uint16_t _e81 = val;
    val = (_e81 - uint16_t(1));
    uint16_t _e84 = val;
    val = (_e84 * uint16_t(2));
    uint16_t _e87 = val;
    val = naga_div(_e87, uint16_t(3));
    uint16_t _e90 = val;
    val = naga_mod(_e90, uint16_t(4));
    uint16_t _e93 = val;
    val = (_e93 & uint16_t(255));
    uint16_t _e96 = val;
    val = (_e96 | uint16_t(16));
    uint16_t _e99 = val;
    val = (_e99 ^ uint16_t(1));
    uint16_t _e104 = val;
    output.Store(0, asuint(uint(_e104)));
    uint16_t _e108 = val;
    output.Store(4, asuint(int(_e108)));
    uint16_t _e112 = val;
    output.Store(8, asuint(float(_e112)));
    uint _e116 = asuint(output.Load(0));
    val = uint16_t(_e116);
    uint16_t _e118 = val;
    local = _e118;
    uint16_t _e120 = local;
    x_1 = int16_t(60);
    val_1 = int16_t(20);
    arr = Constructarray4_int16_t_(int16_t(1), int16_t(2), int16_t(3), int16_t(4));
    int16_t _e127 = val_1;
    val_1 = (_e127 + int16_t(5));
    int16_t _e130 = val_1;
    uint _e133 = input_uniform.val_u32_;
    val_1 = (_e130 + int16_t(_e133));
    int16_t _e136 = val_1;
    int _e139 = input_uniform.val_i32_;
    val_1 = (_e136 + int16_t(_e139));
    int16_t _e142 = val_1;
    int16_t _e145 = input_uniform.val_i16_;
    val_1 = (_e142 + (_e145).xxx.z);
    int16_t _e153 = input_uniform.val_i16_;
    int16_t _e156 = input_storage.Load<int16_t>(40);
    output.Store(40, (_e153 + _e156));
    int16_t2 _e162 = input_uniform.val_i16_2_;
    int16_t2 _e165 = input_storage.Load<int16_t2>(44);
    output.Store(44, (_e162 + _e165));
    int16_t3 _e171 = input_uniform.val_i16_3_;
    int16_t3 _e174 = input_storage.Load<int16_t3>(48);
    output.Store(48, (_e171 + _e174));
    int16_t4 _e180 = input_uniform.val_i16_4_;
    int16_t4 _e183 = input_storage.Load<int16_t4>(56);
    output.Store(56, (_e180 + _e183));
    int16_t _e189[2] = Constructarray2_int16_t_(input_arrays.Load<int16_t>(4+0), input_arrays.Load<int16_t>(4+2));
    {
        int16_t _value2[2] = _e189;
        output_arrays.Store(4+0, _value2[0]);
        output_arrays.Store(4+2, _value2[1]);
    }
    int16_t _e190 = val_1;
    val_1 = abs(_e190);
    int16_t _e192 = val_1;
    int16_t _e193 = val_1;
    val_1 = max(_e192, _e193);
    int16_t _e195 = val_1;
    int16_t _e196 = val_1;
    val_1 = min(_e195, _e196);
    int16_t _e198 = val_1;
    int16_t _e199 = val_1;
    int16_t _e200 = val_1;
    val_1 = clamp(_e198, _e199, _e200);
    int16_t _e202 = val_1;
    val_1 = sign(_e202);
    int16_t _e204 = val_1;
    val_1 = (_e204 - int16_t(1));
    int16_t _e207 = val_1;
    val_1 = (_e207 * int16_t(2));
    int16_t _e210 = val_1;
    val_1 = naga_div(_e210, int16_t(3));
    int16_t _e213 = val_1;
    val_1 = naga_mod(_e213, int16_t(4));
    int16_t _e216 = val_1;
    val_1 = (_e216 & int16_t(255));
    int16_t _e219 = val_1;
    val_1 = (_e219 | int16_t(16));
    int16_t _e222 = val_1;
    val_1 = (_e222 ^ int16_t(1));
    int16_t _e225 = val_1;
    val_1 = (_e225 << 2u);
    int16_t _e228 = val_1;
    val_1 = (_e228 >> 1u);
    int16_t _e231 = val_1;
    val_1 = -(_e231);
    int16_t _e233 = val_1;
    val_1 = ((_e233 < int16_t(0)) ? int16_t(2) : int16_t(1));
    int16_t _e241 = val_1;
    arr[0] = _e241;
    int16_t _e243 = arr[1];
    val_1 = _e243;
    int16_t _e246 = arr[uint16_t(1)];
    val_1 = _e246;
    int16_t _e249 = val_1;
    output.Store(0, asuint(uint(_e249)));
    int16_t _e253 = val_1;
    output.Store(4, asuint(int(_e253)));
    int16_t _e257 = val_1;
    output.Store(8, asuint(float(_e257)));
    uint _e261 = asuint(output.Load(0));
    val_1 = int16_t(_e261);
    int16_t _e263 = val_1;
    val_1 = int16_t(uint16_t(_e263));
    int16_t2 _e268 = input_uniform.val_i16_2_;
    int16_t2 _e271 = input_uniform.val_i16_2_;
    output.Store(44, ((_e268 + _e271) * (int16_t(2)).xx));
    int16_t _e278 = val_1;
    local_1 = _e278;
    int16_t _e280 = local_1;
    output.Store(64, (_e120 + uint16_t(_e280)));
    sg_val = int16_t(subgroup_invocation_id);
    int16_t _e286 = sg_val;
    const int16_t _e309 = WaveActiveSum(_e286);
    sg_val = _e309;
    int16_t _e287 = sg_val;
    const int16_t _e310 = WaveActiveProduct(_e287);
    sg_val = _e310;
    int16_t _e288 = sg_val;
    const int16_t _e311 = WaveActiveMin(_e288);
    sg_val = _e311;
    int16_t _e289 = sg_val;
    const int16_t _e312 = WaveActiveMax(_e289);
    sg_val = _e312;
    int16_t _e290 = sg_val;
    const int16_t _e313 = WavePrefixSum(_e290);
    sg_val = _e313;
    int16_t _e291 = sg_val;
    const int16_t _e314 = _e291 + WavePrefixSum(_e291);
    sg_val = _e314;
    int16_t _e292 = sg_val;
    const int16_t _e315 = WaveReadLaneFirst(_e292);
    sg_val = _e315;
    int16_t _e293 = sg_val;
    const int16_t _e317 = WaveReadLaneAt(_e293, 4u);
    sg_val = _e317;
    sg_uval = uint16_t(subgroup_invocation_id);
    uint16_t _e296 = sg_uval;
    const uint16_t _e318 = WaveActiveSum(_e296);
    sg_uval = _e318;
    uint16_t _e297 = sg_uval;
    const uint16_t _e319 = WaveActiveMin(_e297);
    sg_uval = _e319;
    uint16_t _e298 = sg_uval;
    const uint16_t _e320 = WaveActiveMax(_e298);
    sg_uval = _e320;
    int16_t _e301 = sg_val;
    output.Store(40, _e301);
    uint16_t _e304 = sg_uval;
    output.Store(12, _e304);
    return;
}

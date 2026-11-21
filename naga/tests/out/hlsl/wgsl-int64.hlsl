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

static const uint64_t constant_variable = 20ULL;

static int64_t private_variable = 1LL;
cbuffer input_uniform : register(b0) { UniformCompatible input_uniform; }
ByteAddressBuffer input_storage : register(t1);
ByteAddressBuffer input_arrays : register(t2);
RWByteAddressBuffer output : register(u3);
RWByteAddressBuffer output_arrays : register(u4);

uint64_t naga_f2u64(float value) {
    return uint64_t(clamp(value, 0.0, 1.8446743e19));
}

int64_t naga_f2i64(float value) {
    return int64_t(clamp(value, -9.223372e18, 9.2233715e18));
}

typedef uint64_t ret_Constructarray2_uint64_t_[2];
ret_Constructarray2_uint64_t_ Constructarray2_uint64_t_(uint64_t arg0, uint64_t arg1) {
    uint64_t ret[2] = { arg0, arg1 };
    return ret;
}

typedef int64_t ret_Constructarray2_int64_t_[2];
ret_Constructarray2_int64_t_ Constructarray2_int64_t_(int64_t arg0, int64_t arg1) {
    int64_t ret[2] = { arg0, arg1 };
    return ret;
}

[numthreads(1, 1, 1)]
void main()
{
    uint64_t x = (uint64_t)0;
    uint64_t val = (uint64_t)0;
    uint64_t local = (uint64_t)0;
    int64_t x_1 = (int64_t)0;
    int64_t val_1 = (int64_t)0;
    int64_t local_1 = (int64_t)0;

    x = 67ULL;
    val = 20ULL;
    uint64_t _e3 = val;
    val = (_e3 + ((31ULL + 18446744073709551615ULL) - 18446744073709551615ULL));
    uint64_t _e10 = val;
    uint64_t _e11 = val;
    val = (_e10 + (_e11 + 5ULL));
    uint64_t _e15 = val;
    uint _e18 = input_uniform.val_u32_;
    uint64_t _e19 = val;
    val = (_e15 + uint64_t((_e18 + uint(_e19))));
    uint64_t _e24 = val;
    int _e27 = input_uniform.val_i32_;
    uint64_t _e28 = val;
    val = (_e24 + uint64_t(asint(asuint(_e27) + asuint(int(_e28)))));
    uint64_t _e33 = val;
    float _e36 = input_uniform.val_f32_;
    uint64_t _e37 = val;
    val = (_e33 + naga_f2u64((_e36 + float(_e37))));
    uint64_t _e42 = val;
    uint64_t _e45 = input_uniform.val_u64_;
    val = (_e42 + (_e45).xxx.z);
    uint64_t _e49 = val;
    int64_t _e52 = input_uniform.val_i64_;
    val = (_e49 + uint64_t(_e52));
    uint64_t _e55 = val;
    int64_t2 _e58 = input_uniform.val_i64_2_;
    val = (_e55 + uint64_t2(_e58).y);
    uint64_t _e62 = val;
    int64_t3 _e65 = input_uniform.val_i64_3_;
    val = (_e62 + uint64_t3(_e65).z);
    uint64_t _e69 = val;
    int64_t4 _e72 = input_uniform.val_i64_4_;
    val = (_e69 + uint64_t4(_e72).w);
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
    uint64_t _e117 = val;
    uint64_t _e118 = val;
    val = (_e117 + abs(_e118));
    uint64_t _e121 = val;
    uint64_t _e122 = val;
    uint64_t _e123 = val;
    uint64_t _e124 = val;
    val = (_e121 + clamp(_e122, _e123, _e124));
    uint64_t _e127 = val;
    uint64_t _e128 = val;
    uint64_t _e130 = val;
    val = (_e127 + dot((_e128).xx, (_e130).xx));
    uint64_t _e134 = val;
    uint64_t _e135 = val;
    uint64_t _e136 = val;
    val = (_e134 + max(_e135, _e136));
    uint64_t _e139 = val;
    uint64_t _e140 = val;
    uint64_t _e141 = val;
    val = (_e139 + min(_e140, _e141));
    uint64_t _e144 = val;
    local = _e144;
    uint64_t _e146 = local;
    x_1 = 60LL;
    val_1 = 20LL;
    int64_t _e148 = val_1;
    val_1 = (_e148 + ((31LL - 1002003004005006LL) + -9223372036854775807LL));
    int64_t _e155 = val_1;
    int64_t _e156 = val_1;
    val_1 = (_e155 + (_e156 + 5LL));
    int64_t _e160 = val_1;
    uint _e163 = input_uniform.val_u32_;
    int64_t _e164 = val_1;
    val_1 = (_e160 + int64_t((_e163 + uint(_e164))));
    int64_t _e169 = val_1;
    int _e172 = input_uniform.val_i32_;
    int64_t _e173 = val_1;
    val_1 = (_e169 + int64_t(asint(asuint(_e172) + asuint(int(_e173)))));
    int64_t _e178 = val_1;
    float _e181 = input_uniform.val_f32_;
    int64_t _e182 = val_1;
    val_1 = (_e178 + naga_f2i64((_e181 + float(_e182))));
    int64_t _e187 = val_1;
    int64_t _e190 = input_uniform.val_i64_;
    val_1 = (_e187 + (_e190).xxx.z);
    int64_t _e194 = val_1;
    uint64_t _e197 = input_uniform.val_u64_;
    val_1 = (_e194 + int64_t(_e197));
    int64_t _e200 = val_1;
    uint64_t2 _e203 = input_uniform.val_u64_2_;
    val_1 = (_e200 + int64_t2(_e203).y);
    int64_t _e207 = val_1;
    uint64_t3 _e210 = input_uniform.val_u64_3_;
    val_1 = (_e207 + int64_t3(_e210).z);
    int64_t _e214 = val_1;
    uint64_t4 _e217 = input_uniform.val_u64_4_;
    val_1 = (_e214 + int64_t4(_e217).w);
    int64_t _e221 = val_1;
    val_1 = (_e221 + (-9223372036854775807LL - 1LL));
    int64_t _e228 = input_uniform.val_i64_;
    int64_t _e231 = input_storage.Load<int64_t>(128);
    output.Store(128, (_e228 + _e231));
    int64_t2 _e237 = input_uniform.val_i64_2_;
    int64_t2 _e240 = input_storage.Load<int64_t2>(144);
    output.Store(144, (_e237 + _e240));
    int64_t3 _e246 = input_uniform.val_i64_3_;
    int64_t3 _e249 = input_storage.Load<int64_t3>(160);
    output.Store(160, (_e246 + _e249));
    int64_t4 _e255 = input_uniform.val_i64_4_;
    int64_t4 _e258 = input_storage.Load<int64_t4>(192);
    output.Store(192, (_e255 + _e258));
    int64_t _e264[2] = Constructarray2_int64_t_(input_arrays.Load<int64_t>(16+0), input_arrays.Load<int64_t>(16+8));
    {
        int64_t _value2[2] = _e264;
        output_arrays.Store(16+0, _value2[0]);
        output_arrays.Store(16+8, _value2[1]);
    }
    int64_t _e265 = val_1;
    int64_t _e266 = val_1;
    val_1 = (_e265 + abs(_e266));
    int64_t _e269 = val_1;
    int64_t _e270 = val_1;
    int64_t _e271 = val_1;
    int64_t _e272 = val_1;
    val_1 = (_e269 + clamp(_e270, _e271, _e272));
    int64_t _e275 = val_1;
    int64_t _e276 = val_1;
    int64_t _e278 = val_1;
    val_1 = (_e275 + dot((_e276).xx, (_e278).xx));
    int64_t _e282 = val_1;
    int64_t _e283 = val_1;
    int64_t _e284 = val_1;
    val_1 = (_e282 + max(_e283, _e284));
    int64_t _e287 = val_1;
    int64_t _e288 = val_1;
    int64_t _e289 = val_1;
    val_1 = (_e287 + min(_e288, _e289));
    int64_t _e292 = val_1;
    int64_t _e293 = val_1;
    val_1 = (_e292 + sign(_e293));
    int64_t _e296 = val_1;
    local_1 = _e296;
    int64_t _e298 = local_1;
    output.Store(224, (_e146 + uint64_t(_e298)));
    return;
}

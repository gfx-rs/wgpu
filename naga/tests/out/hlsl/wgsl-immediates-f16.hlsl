struct Inner {
    half h;
    uint16_t _pad1_0;
    int i;
};

struct Data {
    half a;
    half b;
    int _pad2_0;
    half3 v3_;
    uint16_t _pad3_0;
    row_major half2x3 m;
    uint16_t _pad4_0;
    Inner inner;
    float f;
    uint u;
};

cbuffer data_block : register(b0) { uint4 data_raw[3]; };
static Data data;
RWByteAddressBuffer out_ : register(u0);
Inner ConstructInner(half arg0, int arg1) {
    Inner ret = (Inner)0;
    ret.h = arg0;
    ret.i = arg1;
    return ret;
}

Data ConstructData(half arg0, half arg1, half3 arg2, half2x3 arg3, Inner arg4, float arg5, uint arg6) {
    Data ret = (Data)0;
    ret.a = arg0;
    ret.b = arg1;
    ret.v3_ = arg2;
    ret.m = arg3;
    ret.inner = arg4;
    ret.f = arg5;
    ret.u = arg6;
    return ret;
}


[numthreads(1, 1, 1)]
void main(uint i : SV_GroupIndex)
{
    data = ConstructData(asfloat16(uint16_t(data_raw[0].x >> 0u)), asfloat16(uint16_t(data_raw[0].x >> 16u)), half3(asfloat16(uint16_t(data_raw[0].z >> 0u)), asfloat16(uint16_t(data_raw[0].z >> 16u)), asfloat16(uint16_t(data_raw[0].w >> 0u))), half2x3(half3(asfloat16(uint16_t(data_raw[1].x >> 0u)), asfloat16(uint16_t(data_raw[1].x >> 16u)), asfloat16(uint16_t(data_raw[1].y >> 0u))), half3(asfloat16(uint16_t(data_raw[1].z >> 0u)), asfloat16(uint16_t(data_raw[1].z >> 16u)), asfloat16(uint16_t(data_raw[1].w >> 0u)))), ConstructInner(asfloat16(uint16_t(data_raw[2].x >> 0u)), asint(data_raw[2].y)), asfloat(data_raw[2].z), asuint(data_raw[2].w));
    Data whole = data;
    out_.Store(0, asuint(float(whole.a)));
    half _e11 = data.b;
    out_.Store(4, asuint(float(_e11)));
    half _e18 = data.v3_.z;
    out_.Store(8, asuint(float(_e18)));
    half _e25 = data.v3_[min(uint(i), 2u)];
    out_.Store(12, asuint(float(_e25)));
    half3 column = data.m[1];
    out_.Store(16, asuint(float(column.y)));
    half _e41 = data.m[min(uint(i), 1u)].z;
    out_.Store(20, asuint(float(_e41)));
    half _e48 = data.inner.h;
    int _e53 = data.inner.i;
    out_.Store(24, asuint((float(_e48) + float(_e53))));
    float _e60 = data.f;
    uint _e63 = data.u;
    out_.Store(28, asuint((_e60 + float(_e63))));
    return;
}

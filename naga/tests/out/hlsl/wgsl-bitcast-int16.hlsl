struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

int naga_bitcast_int16_t2_to_int(int16_t2 value) {
    uint16_t2 bits = asuint16(value);
    return asint((uint(bits.x) | (uint(bits.y) << 16)));
}

uint naga_bitcast_uint16_t2_to_uint(uint16_t2 value) {
    uint16_t2 bits = asuint16(value);
    return (uint(bits.x) | (uint(bits.y) << 16));
}

float naga_bitcast_uint16_t2_to_float(uint16_t2 value) {
    uint16_t2 bits = asuint16(value);
    return asfloat((uint(bits.x) | (uint(bits.y) << 16)));
}

uint2 naga_bitcast_int16_t4_to_uint2(int16_t4 value) {
    uint16_t4 bits = asuint16(value);
    return uint2((uint(bits.x) | (uint(bits.y) << 16)), (uint(bits.z) | (uint(bits.w) << 16)));
}

float2 naga_bitcast_uint16_t4_to_float2(uint16_t4 value) {
    uint16_t4 bits = asuint16(value);
    return asfloat(uint2((uint(bits.x) | (uint(bits.y) << 16)), (uint(bits.z) | (uint(bits.w) << 16))));
}

int16_t2 naga_bitcast_uint_to_int16_t2(uint value) {
    uint bits = value;
    return int16_t2(uint16_t2(uint16_t(bits & 0xFFFFu), uint16_t(bits >> 16)));
}

uint16_t2 naga_bitcast_float_to_uint16_t2(float value) {
    uint bits = asuint(value);
    return (uint16_t2(uint16_t(bits & 0xFFFFu), uint16_t(bits >> 16)));
}

int16_t4 naga_bitcast_float2_to_int16_t4(float2 value) {
    uint2 bits = asuint(value);
    return int16_t4(uint16_t4(uint16_t(bits.x & 0xFFFFu), uint16_t(bits.x >> 16), uint16_t(bits.y & 0xFFFFu), uint16_t(bits.y >> 16)));
}

uint16_t4 naga_bitcast_int2_to_uint16_t4(int2 value) {
    uint2 bits = asuint(value);
    return (uint16_t4(uint16_t(bits.x & 0xFFFFu), uint16_t(bits.x >> 16), uint16_t(bits.y & 0xFFFFu), uint16_t(bits.y >> 16)));
}

[numthreads(1, 1, 1)]
void main()
{
    int16_t2 i2_ = int16_t2(int16_t(1), int16_t(2));
    uint16_t2 u2_ = uint16_t2(uint16_t(1), uint16_t(2));
    int16_t4 i4_ = int16_t4(int16_t(1), int16_t(2), int16_t(3), int16_t(4));
    uint16_t4 u4_ = uint16_t4(uint16_t(1), uint16_t(2), uint16_t(3), uint16_t(4));
    int i1_ = int(0);
    uint u1_ = 0u;
    float f1_ = 0.0;
    int2 iv2_ = (int(0)).xx;
    uint2 uv2_ = (0u).xx;
    float2 fv2_ = (0.0).xx;

    int16_t2 _e35 = i2_;
    i1_ = naga_bitcast_int16_t2_to_int(_e35);
    uint16_t2 _e37 = u2_;
    u1_ = naga_bitcast_uint16_t2_to_uint(_e37);
    uint16_t2 _e39 = u2_;
    f1_ = naga_bitcast_uint16_t2_to_float(_e39);
    int16_t4 _e41 = i4_;
    uv2_ = naga_bitcast_int16_t4_to_uint2(_e41);
    uint16_t4 _e43 = u4_;
    fv2_ = naga_bitcast_uint16_t4_to_float2(_e43);
    uint _e45 = u1_;
    i2_ = naga_bitcast_uint_to_int16_t2(_e45);
    float _e47 = f1_;
    u2_ = naga_bitcast_float_to_uint16_t2(_e47);
    float2 _e49 = fv2_;
    i4_ = naga_bitcast_float2_to_int16_t4(_e49);
    int2 _e51 = iv2_;
    u4_ = naga_bitcast_int2_to_uint16_t4(_e51);
    return;
}

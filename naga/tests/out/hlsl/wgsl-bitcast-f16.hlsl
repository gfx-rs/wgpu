struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

int naga_bitcast_half2_to_int(half2 value) {
    uint16_t2 bits = asuint16(value);
    return asint((uint(bits.x) | (uint(bits.y) << 16)));
}

uint naga_bitcast_half2_to_uint(half2 value) {
    uint16_t2 bits = asuint16(value);
    return (uint(bits.x) | (uint(bits.y) << 16));
}

float naga_bitcast_half2_to_float(half2 value) {
    uint16_t2 bits = asuint16(value);
    return asfloat((uint(bits.x) | (uint(bits.y) << 16)));
}

int2 naga_bitcast_half4_to_int2(half4 value) {
    uint16_t4 bits = asuint16(value);
    return asint(uint2((uint(bits.x) | (uint(bits.y) << 16)), (uint(bits.z) | (uint(bits.w) << 16))));
}

uint2 naga_bitcast_half4_to_uint2(half4 value) {
    uint16_t4 bits = asuint16(value);
    return uint2((uint(bits.x) | (uint(bits.y) << 16)), (uint(bits.z) | (uint(bits.w) << 16)));
}

float2 naga_bitcast_half4_to_float2(half4 value) {
    uint16_t4 bits = asuint16(value);
    return asfloat(uint2((uint(bits.x) | (uint(bits.y) << 16)), (uint(bits.z) | (uint(bits.w) << 16))));
}

half2 naga_bitcast_int_to_half2(int value) {
    uint bits = asuint(value);
    return asfloat16(uint16_t2(uint16_t(bits & 0xFFFFu), uint16_t(bits >> 16)));
}

half2 naga_bitcast_uint_to_half2(uint value) {
    uint bits = value;
    return asfloat16(uint16_t2(uint16_t(bits & 0xFFFFu), uint16_t(bits >> 16)));
}

half2 naga_bitcast_float_to_half2(float value) {
    uint bits = asuint(value);
    return asfloat16(uint16_t2(uint16_t(bits & 0xFFFFu), uint16_t(bits >> 16)));
}

half4 naga_bitcast_int2_to_half4(int2 value) {
    uint2 bits = asuint(value);
    return asfloat16(uint16_t4(uint16_t(bits.x & 0xFFFFu), uint16_t(bits.x >> 16), uint16_t(bits.y & 0xFFFFu), uint16_t(bits.y >> 16)));
}

half4 naga_bitcast_uint2_to_half4(uint2 value) {
    uint2 bits = value;
    return asfloat16(uint16_t4(uint16_t(bits.x & 0xFFFFu), uint16_t(bits.x >> 16), uint16_t(bits.y & 0xFFFFu), uint16_t(bits.y >> 16)));
}

half4 naga_bitcast_float2_to_half4(float2 value) {
    uint2 bits = asuint(value);
    return asfloat16(uint16_t4(uint16_t(bits.x & 0xFFFFu), uint16_t(bits.x >> 16), uint16_t(bits.y & 0xFFFFu), uint16_t(bits.y >> 16)));
}

[numthreads(1, 1, 1)]
void main()
{
    half2 h2_ = half2(1.0h, 2.0h);
    half4 h4_ = half4(1.0h, 2.0h, 3.0h, 4.0h);
    int i1_ = int(0);
    uint u1_ = 0u;
    float f1_ = 0.0;
    int2 i2_ = (int(0)).xx;
    uint2 u2_ = (0u).xx;
    float2 f2_ = (0.0).xx;

    half2 _e25 = h2_;
    i1_ = naga_bitcast_half2_to_int(_e25);
    half2 _e27 = h2_;
    u1_ = naga_bitcast_half2_to_uint(_e27);
    half2 _e29 = h2_;
    f1_ = naga_bitcast_half2_to_float(_e29);
    half4 _e31 = h4_;
    i2_ = naga_bitcast_half4_to_int2(_e31);
    half4 _e33 = h4_;
    u2_ = naga_bitcast_half4_to_uint2(_e33);
    half4 _e35 = h4_;
    f2_ = naga_bitcast_half4_to_float2(_e35);
    int _e37 = i1_;
    h2_ = naga_bitcast_int_to_half2(_e37);
    uint _e39 = u1_;
    h2_ = naga_bitcast_uint_to_half2(_e39);
    float _e41 = f1_;
    h2_ = naga_bitcast_float_to_half2(_e41);
    int2 _e43 = i2_;
    h4_ = naga_bitcast_int2_to_half4(_e43);
    uint2 _e45 = u2_;
    h4_ = naga_bitcast_uint2_to_half4(_e45);
    float2 _e47 = f2_;
    h4_ = naga_bitcast_float2_to_half4(_e47);
    half2 _e49 = h2_;
    h2_ = _e49;
    uint _e51 = u1_;
    u1_ = _e51;
    u1_ = 4294967295u;
    f1_ = asfloat(int(1));
    i1_ = asint(1.0);
    h2_ = naga_bitcast_int_to_half2(int(1));
    h2_ = half2(1.0h, 2.0h);
    h4_ = naga_bitcast_int2_to_half4(int2(int(1), int(2)));
    u2_ = uint2(4294967295u, 1u);
    return;
}

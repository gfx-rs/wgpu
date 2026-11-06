struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

[numthreads(1, 1, 1)]
void main()
{
    half h = 1.0h;
    int16_t i = int16_t(1);
    uint16_t u = uint16_t(2);
    half4 h4_ = half4(1.0h, 2.0h, 3.0h, 4.0h);
    uint16_t4 u4_ = uint16_t4(uint16_t(1), uint16_t(2), uint16_t(3), uint16_t(4));

    half _e18 = h;
    i = asint16(_e18);
    half _e20 = h;
    u = asuint16(_e20);
    int16_t _e22 = i;
    h = asfloat16(_e22);
    uint16_t _e24 = u;
    h = asfloat16(_e24);
    half4 _e26 = h4_;
    u4_ = asuint16(_e26);
    uint16_t4 _e28 = u4_;
    h4_ = asfloat16(_e28);
    return;
}

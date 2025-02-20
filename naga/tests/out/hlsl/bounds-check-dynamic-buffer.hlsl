struct __dynamic_buffer_offsetsTy0 {
    uint _0;
    uint _1;
};
ConstantBuffer<__dynamic_buffer_offsetsTy0> __dynamic_buffer_offsets0: register(b1, space0);

struct __dynamic_buffer_offsetsTy1 {
    uint _0;
};
ConstantBuffer<__dynamic_buffer_offsetsTy1> __dynamic_buffer_offsets1: register(b2, space0);

struct T {
    uint t;
    int _end_pad_0;
    int _end_pad_1;
    int _end_pad_2;
};

RWByteAddressBuffer in_ : register(u0);
RWByteAddressBuffer out_ : register(u1);
cbuffer in_data_uniform : register(b0) { T in_data_uniform[1]; }
RWByteAddressBuffer in_data_storage_g0_b3_ : register(u2);
RWByteAddressBuffer in_data_storage_g0_b4_ : register(u3);
RWByteAddressBuffer in_data_storage_g1_b0_ : register(u4);

[numthreads(1, 1, 1)]
void main()
{
    uint i = asuint(in_.Load(0));
    uint _e7 = in_data_uniform[min(uint(i), 0u)].t;
    out_.Store(0, asuint(_e7));
    uint _e13 = asuint(in_data_storage_g0_b3_.Load(0+i*16+__dynamic_buffer_offsets0._0));
    out_.Store(4, asuint(_e13));
    uint _e19 = asuint(in_data_storage_g0_b4_.Load(0+i*16+__dynamic_buffer_offsets0._1));
    out_.Store(8, asuint(_e19));
    uint _e25 = asuint(in_data_storage_g1_b0_.Load(0+i*16+__dynamic_buffer_offsets1._0));
    out_.Store(12, asuint(_e25));
    return;
}

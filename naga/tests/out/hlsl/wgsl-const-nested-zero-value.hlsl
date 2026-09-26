static const uint4 composed = uint4(uint2(0u, 0u), 7u, 9u);

RWByteAddressBuffer o : register(u0);
RWByteAddressBuffer v : register(u1);

[numthreads(1, 1, 1)]
void main()
{
    o.Store(0, asuint(0u));
    o.Store(4, asuint(7u));
    o.Store(8, asuint(0u));
    o.Store(12, asuint(asuint(0.0)));
    o.Store(16, asuint(asuint(1.0)));
    o.Store(20, asuint(0u));
    o.Store(24, asuint(9u));
    o.Store(28, asuint(7u));
    v.Store4(0, asuint(composed));
    return;
}

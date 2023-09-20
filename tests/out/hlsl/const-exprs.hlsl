RWByteAddressBuffer out_ : register(u0);
RWByteAddressBuffer out2_ : register(u1);
RWByteAddressBuffer out3_ : register(u2);

[numthreads(1, 1, 1)]
void main()
{
    int2 a = int2(1, 2);
    int2 b = int2(3, 4);
    out_.Store4(0, asuint(int4(4, 3, 2, 1)));
    out2_.Store(0, asuint(2));
    out3_.Store(0, asuint(6));
    return;
}

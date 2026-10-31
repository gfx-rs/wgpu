RWByteAddressBuffer buf : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    uint _e4 = asuint(buf.Load(0));
    buf.Store(4294967295, asuint(_e4));
    uint _e9 = asuint(buf.Load(4294967295));
    buf.Store(0, asuint(_e9));
    return;
}

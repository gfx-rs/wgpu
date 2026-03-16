globallycoherent RWByteAddressBuffer coherent_buf : register(u0);
RWByteAddressBuffer plain_buf : register(u1);

[numthreads(1, 1, 1)]
void main()
{
    uint _e6 = asuint(plain_buf.Load(0+0));
    coherent_buf.Store(0+0, asuint(_e6));
    return;
}

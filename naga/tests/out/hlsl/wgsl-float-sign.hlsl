RWByteAddressBuffer floats : register(u0);
RWByteAddressBuffer ints : register(u1);

[numthreads(1, 1, 1)]
void main()
{
    float _e4 = asfloat(floats.Load(4));
    floats.Store(0, asuint(sign(_e4)));
    float _e8 = asfloat(floats.Load(8));
    float _e11 = asfloat(floats.Load(12));
    float2 v = float2(_e8, _e11);
    float2 signs = sign(v);
    floats.Store(8, asuint(signs.x));
    floats.Store(12, asuint(signs.y));
    int _e24 = asint(ints.Load(4));
    ints.Store(0, asuint(sign(_e24)));
    return;
}

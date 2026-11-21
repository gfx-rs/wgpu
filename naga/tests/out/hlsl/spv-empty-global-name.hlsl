struct type_1 {
    int member;
};

RWByteAddressBuffer unnamed : register(u0);

[numthreads(1, 1, 1)]
void main()
{
    int _e3 = asint(unnamed.Load(0));
    unnamed.Store(0, asuint(asint(asuint(_e3) + asuint(int(1)))));
}

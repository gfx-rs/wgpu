RWByteAddressBuffer out_ : register(u0);
RWByteAddressBuffer out2_ : register(u1);

void swizzle_of_compose()
{
    int2 a = int2(1, 2);
    int2 b = int2(3, 4);
    out_.Store4(0, asuint(int4(4, 3, 2, 1)));
    return;
}

void index_of_compose()
{
    int2 a_1 = int2(1, 2);
    int2 b_1 = int2(3, 4);
    int _expr7 = asint(out2_.Load(0));
    out2_.Store(0, asuint((_expr7 + 2)));
    return;
}

void compose_three_deep()
{
    int _expr2 = asint(out2_.Load(0));
    out2_.Store(0, asuint((_expr2 + 6)));
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    return;
}

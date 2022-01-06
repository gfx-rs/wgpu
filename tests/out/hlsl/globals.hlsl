static const bool Foo_2 = true;

struct Foo {
    float3 v3_;
    float v1_;
};

groupshared float wg[10];
groupshared uint at_1;
ByteAddressBuffer alignment : register(t1);
ByteAddressBuffer dummy : register(t2);

[numthreads(1, 1, 1)]
void main()
{
    float Foo_1 = 1.0;
    bool at = true;

    float _expr8 = asfloat(alignment.Load(12));
    wg[3] = _expr8;
    float _expr13 = asfloat(alignment.Load(0+0));
    wg[2] = _expr13;
    at_1 = 2u;
    return;
}

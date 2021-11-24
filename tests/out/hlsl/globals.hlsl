static const bool Foo_1 = true;

groupshared float wg[10];
groupshared uint at_1;

[numthreads(1, 1, 1)]
void main()
{
    float Foo = 1.0;
    bool at = true;

    wg[3] = 1.0;
    at_1 = 2u;
    return;
}

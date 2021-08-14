static const bool Foo = true;

groupshared float wg[10];
groupshared uint at;

[numthreads(1, 1, 1)]
void main()
{
    wg[3] = 1.0;
    at = 2u;
    return;
}

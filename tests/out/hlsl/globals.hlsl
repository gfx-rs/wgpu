static const bool Foo = true;

groupshared float wg[10];

[numthreads(1, 1, 1)]
void main()
{
    wg[3] = 1.0;
    return;
}

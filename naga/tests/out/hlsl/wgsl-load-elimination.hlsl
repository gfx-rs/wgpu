static uint sink = 0u;

void simple()
{
    uint a = (uint)0;

    uint _e1 = sink;
    a = _e1;
    uint b = a;
    a = 2u;
    sink = b;
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    simple();
    return;
}

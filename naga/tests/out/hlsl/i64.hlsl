static const uint k = 20uL;

static int v = 1L;

int fi(int x)
{
    int z = (int)0;

    int y = (31L - 1002003004005006L);
    z = (y + 5L);
    return (((x + y) + 20L) + 50L);
}

uint fu(uint x_1)
{
    uint z_1 = (uint)0;

    uint y_1 = (31uL + 1002003004005006uL);
    z_1 = (y_1 + 4uL);
    return (((x_1 + y_1) + k) + 34uL);
}

[numthreads(1, 1, 1)]
void main()
{
    const uint _e1 = fu(67uL);
    const int _e3 = fi(60L);
    return;
}

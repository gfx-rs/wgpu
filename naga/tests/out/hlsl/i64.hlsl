static const uint64_t k = 20uL;

static int64_t v = 1L;

int64_t fi(int64_t x)
{
    int64_t z = (int64_t)0;

    int64_t y = (31L - 1002003004005006L);
    z = (y + 5L);
    return (((x + y) + 20L) + 50L);
}

uint64_t fu(uint64_t x_1)
{
    uint64_t z_1 = (uint64_t)0;

    uint64_t y_1 = (31uL + 1002003004005006uL);
    uint64_t3 v_1 = uint64_t3(3uL, 4uL, 5uL);
    z_1 = (y_1 + 4uL);
    return ((((((x_1 + y_1) + k) + 34uL) + v_1.x) + v_1.y) + v_1.z);
}

[numthreads(1, 1, 1)]
void main()
{
    const uint64_t _e1 = fu(67uL);
    const int64_t _e3 = fi(60L);
    return;
}

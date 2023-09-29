static const uint TWO = 2u;
static const int THREE = 3;
static const int FOUR = 4;
static const int FOUR_ALIAS = 4;
static const int TEST_CONSTANT_ADDITION = 8;
static const int TEST_CONSTANT_ALIAS_ADDITION = 8;

RWByteAddressBuffer out_ : register(u0);
RWByteAddressBuffer out2_ : register(u1);

void swizzle_of_compose()
{
    out_.Store4(0, asuint(int4(4, 3, 2, 1)));
    return;
}

void index_of_compose()
{
    int _expr2 = asint(out2_.Load(0));
    out2_.Store(0, asuint((_expr2 + 2)));
    return;
}

void compose_three_deep()
{
    int _expr2 = asint(out2_.Load(0));
    out2_.Store(0, asuint((_expr2 + 6)));
    return;
}

void non_constant_initializers()
{
    int w = 30;
    int x = (int)0;
    int y = (int)0;
    int z = 70;

    int _expr2 = w;
    x = _expr2;
    int _expr4 = x;
    y = _expr4;
    int _expr9 = w;
    int _expr10 = x;
    int _expr11 = y;
    int _expr12 = z;
    int4 _expr14 = asint(out_.Load4(0));
    out_.Store4(0, asuint((_expr14 + int4(_expr9, _expr10, _expr11, _expr12))));
    return;
}

void splat_of_constant()
{
    out_.Store4(0, asuint(int4(-4, -4, -4, -4)));
    return;
}

void compose_of_constant()
{
    out_.Store4(0, asuint(int4(-4, -4, -4, -4)));
    return;
}

[numthreads(2, 3, 1)]
void main()
{
    swizzle_of_compose();
    index_of_compose();
    compose_three_deep();
    non_constant_initializers();
    splat_of_constant();
    compose_of_constant();
    return;
}

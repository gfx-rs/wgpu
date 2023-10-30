static const uint TWO = 2u;
static const int THREE = 3;
static const int FOUR = 4;
static const int FOUR_ALIAS = 4;
static const int TEST_CONSTANT_ADDITION = 8;
static const int TEST_CONSTANT_ALIAS_ADDITION = 8;
static const float PI = 3.141;
static const float phi_sun = 6.282;
static const float4 DIV = float4(0.44444445, 0.0, 0.0, 0.0);
static const int TEXTURE_KIND_REGULAR = 0;
static const int TEXTURE_KIND_WARP = 1;
static const int TEXTURE_KIND_SKY = 2;

void swizzle_of_compose()
{
    int4 out_ = int4(4, 3, 2, 1);

}

void index_of_compose()
{
    int out_1 = 2;

}

void compose_three_deep()
{
    int out_2 = 6;

}

void non_constant_initializers()
{
    int w = 30;
    int x = (int)0;
    int y = (int)0;
    int z = 70;
    int4 out_3 = (int4)0;

    int _expr2 = w;
    x = _expr2;
    int _expr4 = x;
    y = _expr4;
    int _expr8 = w;
    int _expr9 = x;
    int _expr10 = y;
    int _expr11 = z;
    out_3 = int4(_expr8, _expr9, _expr10, _expr11);
    return;
}

void splat_of_constant()
{
    int4 out_4 = int4(-4, -4, -4, -4);

}

void compose_of_constant()
{
    int4 out_5 = int4(-4, -4, -4, -4);

}

uint map_texture_kind(int texture_kind)
{
    switch(texture_kind) {
        case 0: {
            return 10u;
        }
        case 1: {
            return 20u;
        }
        case 2: {
            return 30u;
        }
        default: {
            return 0u;
        }
    }
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

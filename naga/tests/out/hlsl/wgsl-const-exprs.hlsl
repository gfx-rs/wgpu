static const uint TWO = 2u;
static const int THREE = int(3);
static const bool TRUE = true;
static const bool FALSE = false;
static const int FOUR = int(4);
static const int TEXTURE_KIND_REGULAR = int(0);
static const int TEXTURE_KIND_WARP = int(1);
static const int TEXTURE_KIND_SKY = int(2);
static const int FOUR_ALIAS = int(4);
static const int TEST_CONSTANT_ADDITION = int(8);
static const int TEST_CONSTANT_ALIAS_ADDITION = int(8);
static const float PI = 3.141;
static const float phi_sun = 6.282;
static const float4 DIV = float4(0.44444445, 0.0, 0.0, 0.0);
static const float2 add_vec = float2(4.0, 5.0);
static const bool2 compare_vec = bool2(true, false);

void swizzle_of_compose()
{
    int4 out_ = int4(int(4), int(3), int(2), int(1));

    return;
}

void index_of_compose()
{
    int out_1 = int(2);

    return;
}

void compose_three_deep()
{
    int out_2 = int(6);

    return;
}

void non_constant_initializers()
{
    int w = int(30);
    int x = (int)0;
    int y = (int)0;
    int z = int(70);
    int4 out_3 = (int4)0;

    int _e2 = w;
    x = _e2;
    int _e4 = x;
    y = _e4;
    int _e8 = w;
    int _e9 = x;
    int _e10 = y;
    int _e11 = z;
    out_3 = int4(_e8, _e9, _e10, _e11);
    return;
}

void splat_of_constant()
{
    int4 out_4 = int4(int(-4), int(-4), int(-4), int(-4));

    return;
}

void compose_of_constant()
{
    int4 out_5 = int4(int(-4), int(-4), int(-4), int(-4));

    return;
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

void compose_of_splat()
{
    float4 x_1 = float4(2.0, 1.0, 1.0, 1.0);

    return;
}

void test_local_const()
{
    float arr[2] = (float[2])0;

    return;
}

void compose_vector_zero_val_binop()
{
    int3 a = int3(int(1), int(1), int(1));
    int3 b = int3(int(0), int(1), int(2));
    int3 c = int3(int(1), int(0), int(2));

    return;
}

void relational()
{
    bool scalar_any_false = false;
    bool scalar_any_true = true;
    bool scalar_all_false = false;
    bool scalar_all_true = true;
    bool vec_any_false = false;
    bool vec_any_true = true;
    bool vec_all_false = false;
    bool vec_all_true = true;

    return;
}

void packed_dot_product()
{
    int signed_four = int(4);
    uint unsigned_four = 4u;
    int signed_twelve = int(12);
    uint unsigned_twelve = 12u;
    int signed_seventy = int(70);
    uint unsigned_seventy = 70u;
    int minus_four = int(-4);

    return;
}

typedef int ret_Constructarray9_int_[9];
ret_Constructarray9_int_ Constructarray9_int_(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8) {
    int ret[9] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 };
    return ret;
}

void abstract_access(uint i)
{
    float a_1 = 1.0;
    uint b_1 = 1u;
    int c_1 = (int)0;
    int d = (int)0;

    c_1 = Constructarray9_int_(int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9))[min(uint(i), 8u)];
    d = int4(int(1), int(2), int(3), int(4))[min(uint(i), 3u)];
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
    const uint _e1 = map_texture_kind(int(1));
    compose_of_splat();
    test_local_const();
    compose_vector_zero_val_binop();
    relational();
    packed_dot_product();
    test_local_const();
    abstract_access(1u);
    return;
}

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

typedef int ret_Constructarray9_int_[9];
ret_Constructarray9_int_ Constructarray9_int_(int arg0, int arg1, int arg2, int arg3, int arg4, int arg5, int arg6, int arg7, int arg8) {
    int ret[9] = { arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8 };
    return ret;
}

int ZeroValueint() {
    return (int)0;
}

int4 ZeroValueint4() {
    return (int4)0;
}

typedef float ret_ZeroValuearray2_float_[2];
ret_ZeroValuearray2_float_ ZeroValuearray2_float_() {
    return (float[2])0;
}

[numthreads(2, 3, 1)]
void main()
{
    int4 out_ = (int4)0;
    int out_1 = (int)0;
    int out_2 = (int)0;
    int w = (int)0;
    int x = (int)0;
    int y = (int)0;
    int z = (int)0;
    int4 out_3 = (int4)0;
    int4 out_4 = (int4)0;
    int4 out_5 = (int4)0;
    int texture_kind = (int)0;
    uint local = (uint)0;
    bool local_1 = (bool)0;
    float4 x_1 = (float4)0;
    float arr[2] = (float[2])0;
    int3 a = (int3)0;
    int3 b = (int3)0;
    int3 c = (int3)0;
    bool scalar_any_false = (bool)0;
    bool scalar_any_true = (bool)0;
    bool scalar_all_false = (bool)0;
    bool scalar_all_true = (bool)0;
    bool vec_any_false = (bool)0;
    bool vec_any_true = (bool)0;
    bool vec_all_false = (bool)0;
    bool vec_all_true = (bool)0;
    int signed_four = (int)0;
    uint unsigned_four = (uint)0;
    int signed_twelve = (int)0;
    uint unsigned_twelve = (uint)0;
    int signed_seventy = (int)0;
    uint unsigned_seventy = (uint)0;
    int minus_four = (int)0;
    float arr_1[2] = (float[2])0;
    uint i = (uint)0;
    float a_1 = (float)0;
    uint b_1 = (uint)0;
    int c_1 = (int)0;
    int d = (int)0;

    out_ = int4(int(4), int(3), int(2), int(1));
    out_1 = int(2);
    out_2 = int(6);
    w = int(30);
    x = ZeroValueint();
    y = ZeroValueint();
    z = int(70);
    out_3 = ZeroValueint4();
    int _e14 = w;
    x = _e14;
    int _e16 = x;
    y = _e16;
    int _e20 = w;
    int _e21 = x;
    int _e22 = y;
    int _e23 = z;
    out_3 = int4(_e20, _e21, _e22, _e23);
    out_4 = int4(int(-4), int(-4), int(-4), int(-4));
    out_5 = int4(int(-4), int(-4), int(-4), int(-4));
    texture_kind = int(1);
    int _e47 = texture_kind;
    do {
        local_1 = false;
        switch(_e47) {
            case 0: {
                local = 10u;
                local_1 = true;
                break;
            }
            case 1: {
                local = 20u;
                local_1 = true;
                break;
            }
            case 2: {
                local = 30u;
                local_1 = true;
                break;
            }
            default: {
                local = 0u;
                local_1 = true;
                break;
            }
        }
        bool _e56 = local_1;
        if (_e56) {
            break;
        }
    } while(false);
    x_1 = float4(2.0, 1.0, 1.0, 1.0);
    arr = ZeroValuearray2_float_();
    a = int3(int(1), int(1), int(1));
    b = int3(int(0), int(1), int(2));
    c = int3(int(1), int(0), int(2));
    scalar_any_false = false;
    scalar_any_true = true;
    scalar_all_false = false;
    scalar_all_true = true;
    vec_any_false = false;
    vec_any_true = true;
    vec_all_false = false;
    vec_all_true = true;
    signed_four = int(4);
    unsigned_four = 4u;
    signed_twelve = int(12);
    unsigned_twelve = 12u;
    signed_seventy = int(70);
    unsigned_seventy = 70u;
    minus_four = int(-4);
    arr_1 = ZeroValuearray2_float_();
    i = 1u;
    uint _e112 = i;
    a_1 = 1.0;
    b_1 = 1u;
    c_1 = ZeroValueint();
    d = ZeroValueint();
    c_1 = Constructarray9_int_(int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9))[min(uint(_e112), 8u)];
    d = int4(int(1), int(2), int(3), int(4))[min(uint(_e112), 3u)];
    return;
}

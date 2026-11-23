typedef int ret_Constructarray4_int_[4];
ret_Constructarray4_int_ Constructarray4_int_(int arg0, int arg1, int arg2, int arg3) {
    int ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

typedef int ret_ZeroValuearray4_int_[4];
ret_ZeroValuearray4_int_ ZeroValuearray4_int_() {
    return (int[4])0;
}

[numthreads(1, 1, 1)]
void main()
{
    int2 vec[4] = (int2[4])0;
    float2x2 mat[4] = (float2x2[4])0;
    int arr1d[4] = (int[4])0;
    int arr2d[4][4] = (int[4][4])0;
    int arr3d[4][4][4] = (int[4][4][4])0;
    uint i = (uint)0;
    int arr[4] = (int[4])0;
    uint i_1 = (uint)0;
    uint i_2 = (uint)0;
    uint i_3 = (uint)0;
    uint j = (uint)0;
    uint i_4 = (uint)0;
    uint j_1 = (uint)0;
    uint i_5 = (uint)0;
    int a[4] = (int[4])0;
    uint i_6 = (uint)0;
    int b[4] = (int[4])0;
    uint i_7 = (uint)0;

    i = 1u;
    arr = Constructarray4_int_(int(1), int(2), int(3), int(4));
    i_1 = 1u;
    i_2 = 1u;
    i_3 = 1u;
    j = 2u;
    i_4 = 1u;
    j_1 = 2u;
    i_5 = 1u;
    a = Constructarray4_int_(int(1), int(2), int(3), int(4));
    int _e30[4] = a;
    i_6 = 5u;
    b = ZeroValuearray4_int_();
    b = _e30;
    i_7 = 1u;
    return;
}

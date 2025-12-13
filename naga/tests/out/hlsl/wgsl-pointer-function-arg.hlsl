void takes_ptr(inout int p)
{
    return;
}

void takes_array_ptr(inout int p_1[4])
{
    return;
}

void takes_vec_ptr(inout int2 p_2)
{
    return;
}

void takes_mat_ptr(inout float2x2 p_3)
{
    return;
}

typedef int ret_Constructarray4_int_[4];
ret_Constructarray4_int_ Constructarray4_int_(int arg0, int arg1, int arg2, int arg3) {
    int ret[4] = { arg0, arg1, arg2, arg3 };
    return ret;
}

void local_var(uint i)
{
    int arr[4] = Constructarray4_int_(int(1), int(2), int(3), int(4));

    takes_ptr(arr[min(uint(i), 3u)]);
    takes_array_ptr(arr);
    return;
}

void mat_vec_ptrs(inout int2 pv[4], inout float2x2 pm[4], uint i_1)
{
    takes_vec_ptr(pv[min(uint(i_1), 3u)]);
    takes_mat_ptr(pm[min(uint(i_1), 3u)]);
    return;
}

void argument(inout int v[4], uint i_2)
{
    takes_ptr(v[min(uint(i_2), 3u)]);
    return;
}

void argument_nested_x2_(inout int v_1[4][4], uint i_3, uint j)
{
    takes_ptr(v_1[min(uint(i_3), 3u)][min(uint(j), 3u)]);
    takes_ptr(v_1[min(uint(i_3), 3u)][0]);
    takes_ptr(v_1[0][min(uint(j), 3u)]);
    takes_array_ptr(v_1[min(uint(i_3), 3u)]);
    return;
}

void argument_nested_x3_(inout int v_2[4][4][4], uint i_4, uint j_1)
{
    takes_ptr(v_2[min(uint(i_4), 3u)][0][min(uint(j_1), 3u)]);
    takes_ptr(v_2[min(uint(i_4), 3u)][min(uint(j_1), 3u)][0]);
    takes_ptr(v_2[0][min(uint(i_4), 3u)][min(uint(j_1), 3u)]);
    return;
}

void index_from_self(inout int v_3[4], uint i_5)
{
    int _e3 = v_3[min(uint(i_5), 3u)];
    takes_ptr(v_3[min(uint(_e3), 3u)]);
    return;
}

void local_var_from_arg(int a[4], uint i_6)
{
    int b[4] = (int[4])0;

    b = a;
    takes_ptr(b[min(uint(i_6), 3u)]);
    return;
}

void let_binding(inout int a_1[4], uint i_7)
{
    takes_ptr(a_1[min(uint(i_7), 3u)]);
    takes_ptr(a_1[0]);
    return;
}

[numthreads(1, 1, 1)]
void main()
{
    int2 vec[4] = (int2[4])0;
    float2x2 mat[4] = (float2x2[4])0;
    int arr1d[4] = (int[4])0;
    int arr2d[4][4] = (int[4][4])0;
    int arr3d[4][4][4] = (int[4][4][4])0;

    local_var(1u);
    mat_vec_ptrs(vec, mat, 1u);
    argument(arr1d, 1u);
    argument_nested_x2_(arr2d, 1u, 2u);
    argument_nested_x3_(arr3d, 1u, 2u);
    index_from_self(arr1d, 1u);
    local_var_from_arg(Constructarray4_int_(int(1), int(2), int(3), int(4)), 5u);
    let_binding(arr1d, 1u);
    return;
}

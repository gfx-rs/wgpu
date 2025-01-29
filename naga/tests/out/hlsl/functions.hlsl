float2 test_fma()
{
    float2 a = float2(2.0, 2.0);
    float2 b = float2(0.5, 0.5);
    float2 c = float2(0.5, 0.5);
    return mad(a, b, c);
}

int test_integer_dot_product()
{
    int2 a_2_ = (1).xx;
    int2 b_2_ = (1).xx;
    int c_2_ = dot(a_2_, b_2_);
    uint3 a_3_ = (1u).xxx;
    uint3 b_3_ = (1u).xxx;
    uint c_3_ = dot(a_3_, b_3_);
    int c_4_ = dot((4).xxxx, (2).xxxx);
    return c_4_;
}

[numthreads(1, 1, 1)]
void main()
{
    const float2 _e0 = test_fma();
    const int _e1 = test_integer_dot_product();
    return;
}


float2 test_fma()
{
    float2 a = float2(2.0, 2.0);
    float2 b = float2(0.5, 0.5);
    float2 c = float2(0.5, 0.5);
    return mad(a, b, c);
}

[numthreads(1, 1, 1)]
void main()
{
    const float2 _e0 = test_fma();
    return;
}

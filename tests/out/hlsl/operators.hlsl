float4 splat()
{
    float2 a = (((float2(1.0.xx) + float2(2.0.xx)) - float2(3.0.xx)) / float2(4.0.xx));
    int4 b = (int4(5.xxxx) % int4(2.xxxx));
    return (a.xyxy + float4(b));
}

int unary()
{
    if ((!true)) {
        return 1;
    } else {
        return (!bool(1));
    }
}

[numthreads(1, 1, 1)]
void main()
{
    const float4 _e0 = splat();
    const int _e1 = unary();
    return;
}

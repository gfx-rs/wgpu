ByteAddressBuffer asdf : register(t0);

float compute()
{
    float _e1 = asfloat(asdf.Load(0));
    float u03b8_2_ = (_e1 + 9001.0);
    return u03b8_2_;
}

[numthreads(1, 1, 1)]
void main()
{
    const float _e0 = compute();
    return;
}

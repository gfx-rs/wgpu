ByteAddressBuffer asdf : register(t0);

[numthreads(1, 1, 1)]
void main()
{
    float local = (float)0;

    float _e1 = asfloat(asdf.Load(0));
    local = (_e1 + 9001.0);
    return;
}

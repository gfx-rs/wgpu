[numthreads(1, 1, 1)]
void main()
{
    int2 x0_ = int2(int(1), int(2));
    float2 i1_ = (float2)0;

    int _e12 = x0_.x;
    int _e14 = x0_.y;
    i1_ = ((_e12 < _e14) ? float2(0.0, 1.0) : float2(1.0, 0.0));
    return;
}

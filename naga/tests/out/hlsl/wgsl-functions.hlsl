[numthreads(1, 1, 1)]
void main()
{
    float2 local = (float2)0;
    int local_1 = (int)0;
    uint local_2 = (uint)0;

    local = mad(float2(2.0, 2.0), float2(0.5, 0.5), float2(0.5, 0.5));
    float2 a = local;
    local_1 = int(32);
    int b = local_1;
    uint _e16 = dot(uint4(3u, 3u >> 8, 3u >> 16, 3u >> 24) << 24 >> 24, uint4(4u, 4u >> 8, 4u >> 16, 4u >> 24) << 24 >> 24);
    uint _e18 = (7u + _e16);
    uint _e20 = (8u + _e16);
    local_2 = dot(uint4(_e18, _e18 >> 8, _e18 >> 16, _e18 >> 24) << 24 >> 24, uint4(_e20, _e20 >> 8, _e20 >> 16, _e20 >> 24) << 24 >> 24);
    uint c = local_2;
    return;
}

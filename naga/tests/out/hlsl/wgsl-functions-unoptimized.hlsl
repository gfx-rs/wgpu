[numthreads(1, 1, 1)]
void main()
{
    uint local = (uint)0;

    uint _e2 = dot(uint4(3u, 3u >> 8, 3u >> 16, 3u >> 24) << 24 >> 24, uint4(4u, 4u >> 8, 4u >> 16, 4u >> 24) << 24 >> 24);
    uint _e4 = (7u + _e2);
    uint _e6 = (8u + _e2);
    local = dot(uint4(_e4, _e4 >> 8, _e4 >> 16, _e4 >> 24) << 24 >> 24, uint4(_e6, _e6 >> 8, _e6 >> 16, _e6 >> 24) << 24 >> 24);
    uint c = local;
    return;
}

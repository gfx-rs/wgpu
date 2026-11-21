[numthreads(1, 1, 1)]
void main()
{
    uint local = (uint)0;

    uint _e2 = dot4add_u8packed(3u, 4u, 0);
    uint _e4 = (7u + _e2);
    uint _e6 = (8u + _e2);
    local = dot4add_u8packed(_e4, _e6, 0);
    uint c = local;
    return;
}

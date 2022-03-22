
RWByteAddressBuffer v_indices : register(u0);

uint collatz_iterations(uint n_base)
{
    uint n = (uint)0;
    uint i = 0u;

    n = n_base;
    while(true) {
        uint _expr5 = n;
        if ((_expr5 > 1u)) {
        } else {
            break;
        }
        uint _expr8 = n;
        if (((_expr8 % 2u) == 0u)) {
            uint _expr13 = n;
            n = (_expr13 / 2u);
        } else {
            uint _expr17 = n;
            n = ((3u * _expr17) + 1u);
        }
        uint _expr21 = i;
        i = (_expr21 + 1u);
    }
    uint _expr24 = i;
    return _expr24;
}

[numthreads(1, 1, 1)]
void main(uint3 global_id : SV_DispatchThreadID)
{
    uint _expr8 = asuint(v_indices.Load(global_id.x*4+0));
    const uint _e9 = collatz_iterations(_expr8);
    v_indices.Store(global_id.x*4+0, asuint(_e9));
    return;
}

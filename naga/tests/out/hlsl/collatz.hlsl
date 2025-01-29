RWByteAddressBuffer v_indices : register(u0);

uint collatz_iterations(uint n_base)
{
    uint n = (uint)0;
    uint i = 0u;

    n = n_base;
    while(true) {
        uint _e4 = n;
        if ((_e4 > 1u)) {
        } else {
            break;
        }
        {
            uint _e7 = n;
            if (((_e7 % 2u) == 0u)) {
                uint _e12 = n;
                n = (_e12 / 2u);
            } else {
                uint _e16 = n;
                n = ((3u * _e16) + 1u);
            }
            uint _e20 = i;
            i = (_e20 + 1u);
        }
    }
    uint _e23 = i;
    return _e23;
}

[numthreads(1, 1, 1)]
void main(uint3 global_id : SV_DispatchThreadID)
{
    uint _e9 = asuint(v_indices.Load(global_id.x*4+0));
    const uint _e10 = collatz_iterations(_e9);
    v_indices.Store(global_id.x*4+0, asuint(_e10));
    return;
}

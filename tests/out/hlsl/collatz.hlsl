RWByteAddressBuffer v_indices : register(u0);

uint collatz_iterations(uint n_base)
{
    uint n = (uint)0;
    uint i = 0u;

    n = n_base;
    while(true) {
        uint _expr4 = n;
        if ((_expr4 > 1u)) {
        } else {
            break;
        }
        {
            uint _expr7 = n;
            if (((_expr7 % 2u) == 0u)) {
                uint _expr12 = n;
                n = (_expr12 / 2u);
            } else {
                uint _expr16 = n;
                n = ((3u * _expr16) + 1u);
            }
            uint _expr20 = i;
            i = (_expr20 + 1u);
        }
    }
    uint _expr23 = i;
    return _expr23;
}

[numthreads(1, 1, 1)]
void main(uint3 global_id : SV_DispatchThreadID)
{
    uint _expr9 = asuint(v_indices.Load(global_id.x*4+0));
    const uint _e10 = collatz_iterations(_expr9);
    v_indices.Store(global_id.x*4+0, asuint(_e10));
    return;
}

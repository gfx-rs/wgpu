RWByteAddressBuffer v_indices : register(u0);

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

uint ZeroValueuint() {
    return (uint)0;
}

[numthreads(1, 1, 1)]
void main(uint3 global_id : SV_DispatchThreadID)
{
    uint n_base = (uint)0;
    uint n = (uint)0;
    uint i = (uint)0;
    uint local = (uint)0;

    uint _e9 = asuint(v_indices.Load(global_id.x*4+0));
    n_base = _e9;
    uint _e11 = n_base;
    n = ZeroValueuint();
    i = 0u;
    n = _e11;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e13 = n;
        if ((_e13 > 1u)) {
        } else {
            break;
        }
        {
            uint _e16 = n;
            if ((naga_mod(_e16, 2u) == 0u)) {
                uint _e21 = n;
                n = naga_div(_e21, 2u);
            } else {
                uint _e25 = n;
                n = ((3u * _e25) + 1u);
            }
            uint _e30 = i;
            i = (_e30 + 1u);
        }
    }
    uint _e33 = i;
    local = _e33;
    uint _e35 = local;
    v_indices.Store(global_id.x*4+0, asuint(_e35));
    return;
}

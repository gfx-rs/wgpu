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
    uint2 loop_bound = (uint2)0;

    uint _e9 = asuint(v_indices.Load(global_id.x*4+0));
    n_base = _e9;
    uint _e11 = n_base;
    n = ZeroValueuint();
    i = 0u;
    n = _e11;
    loop_bound = (4294967295u).xx;
    while(true) {
        uint2 _e16 = loop_bound;
        if (all((_e16 == (0u).xx))) {
            break;
        }
        loop_bound = (_e16 - uint2(((_e16.y == 0u) ? 1u : 0u), 1u));
        uint _e27 = n;
        if ((_e27 > 1u)) {
        } else {
            break;
        }
        {
            uint _e30 = n;
            if ((naga_mod(_e30, 2u) == 0u)) {
                uint _e35 = n;
                n = naga_div(_e35, 2u);
            } else {
                uint _e39 = n;
                n = ((3u * _e39) + 1u);
            }
            uint _e44 = i;
            i = (_e44 + 1u);
        }
    }
    uint _e47 = i;
    local = _e47;
    uint _e49 = local;
    v_indices.Store(global_id.x*4+0, asuint(_e49));
    return;
}

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
    uint2 loop_bound = (4294967295u).xx;

    uint _e9 = asuint(v_indices.Load(global_id.x*4+0));
    n_base = _e9;
    uint _e11 = n_base;
    n = ZeroValueuint();
    i = 0u;
    n = _e11;
    while(true) {
        {
            uint2 _e15 = loop_bound;
            loop_bound = (_e15 - uint2(((_e15.y == 0u) ? 1u : 0u), 1u));
            if (all((_e15 == (0u).xx))) {
                break;
            }
        }
        uint _e25 = n;
        if ((_e25 > 1u)) {
        } else {
            break;
        }
        {
            uint _e28 = n;
            if ((naga_mod(_e28, 2u) == 0u)) {
                uint _e33 = n;
                n = naga_div(_e33, 2u);
            } else {
                uint _e37 = n;
                n = ((3u * _e37) + 1u);
            }
            uint _e42 = i;
            i = (_e42 + 1u);
        }
    }
    uint _e45 = i;
    local = _e45;
    uint _e47 = local;
    v_indices.Store(global_id.x*4+0, asuint(_e47));
    return;
}

struct ComputeInput_main {
};

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

uint lowered(uint sid_1)
{
    uint v = 0u;

    switch(naga_mod(sid_1, 3u)) {
        case 0u: {
            const uint _e5 = WaveReadLaneFirst(sid_1);
            v = _e5;
            break;
        }
        case 1u: {
            const uint _e6 = WaveActiveSum(sid_1);
            v = _e6;
            break;
        }
        default: {
            const uint _e7 = WaveReadLaneFirst(sid_1);
            v = _e7;
            break;
        }
    }
    uint _e8 = v;
    return _e8;
}

uint kept_fallthrough(uint sid_2)
{
    uint v_1 = 0u;

    do {
        const uint _e5 = WaveReadLaneFirst(sid_2);
        v_1 = _e5;
    } while(false);
    uint _e6 = v_1;
    return _e6;
}

uint kept_break(uint sid_3)
{
    uint v_2 = 0u;

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        bool should_continue = false;
        switch(naga_mod(sid_3, 2u)) {
            case 0u: {
                const uint _e5 = WaveReadLaneFirst(sid_3);
                v_2 = _e5;
                break;
            }
            default: {
                v_2 = 1u;
                break;
            }
        }
        break;
    }
    uint _e7 = v_2;
    return _e7;
}

uint kept_no_subgroup_ops(uint sid_4)
{
    uint v_3 = 0u;

    switch(naga_mod(sid_4, 2u)) {
        case 0u: {
            v_3 = 10u;
            break;
        }
        default: {
            v_3 = 20u;
            break;
        }
    }
    uint _e7 = v_3;
    return _e7;
}

[numthreads(32, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint sid = WaveGetLaneIndex();
    const uint _e1 = lowered(sid);
    const uint _e2 = kept_fallthrough(sid);
    const uint _e3 = kept_break(sid);
    const uint _e4 = kept_no_subgroup_ops(sid);
    return;
}

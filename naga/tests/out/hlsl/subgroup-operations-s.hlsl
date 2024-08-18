static uint num_subgroups_1 = (uint)0;
static uint subgroup_id_1 = (uint)0;
static uint subgroup_size_1 = (uint)0;
static uint subgroup_invocation_id_1 = (uint)0;

struct ComputeInput_main {
    uint __local_invocation_index : SV_GroupIndex;
};

void main_1()
{
    uint _e5 = subgroup_size_1;
    uint _e6 = subgroup_invocation_id_1;
    const uint4 _e9 = WaveActiveBallot(((_e6 & 1u) == 1u));
    const uint4 _e10 = WaveActiveBallot(true);
    const bool _e12 = WaveActiveAllTrue((_e6 != 0u));
    const bool _e14 = WaveActiveAnyTrue((_e6 == 0u));
    const uint _e15 = WaveActiveSum(_e6);
    const uint _e16 = WaveActiveProduct(_e6);
    const uint _e17 = WaveActiveMin(_e6);
    const uint _e18 = WaveActiveMax(_e6);
    const uint _e19 = WaveActiveBitAnd(_e6);
    const uint _e20 = WaveActiveBitOr(_e6);
    const uint _e21 = WaveActiveBitXor(_e6);
    const uint _e22 = WavePrefixSum(_e6);
    const uint _e23 = WavePrefixProduct(_e6);
    const uint _e24 = _e6 + WavePrefixSum(_e6);
    const uint _e25 = _e6 * WavePrefixProduct(_e6);
    const uint _e26 = WaveReadLaneFirst(_e6);
    const uint _e27 = WaveReadLaneAt(_e6, 4u);
    const uint _e30 = WaveReadLaneAt(_e6, ((_e5 - 1u) - _e6));
    const uint _e31 = WaveReadLaneAt(_e6, WaveGetLaneIndex() + 1u);
    const uint _e32 = WaveReadLaneAt(_e6, WaveGetLaneIndex() - 1u);
    const uint _e34 = WaveReadLaneAt(_e6, WaveGetLaneIndex() ^ (_e5 - 1u));
    return;
}

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint num_subgroups = (1u + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
    uint subgroup_id = computeinput_main.__local_invocation_index / WaveGetLaneCount();
    uint subgroup_size = WaveGetLaneCount();
    uint subgroup_invocation_id = WaveGetLaneIndex();
    num_subgroups_1 = num_subgroups;
    subgroup_id_1 = subgroup_id;
    subgroup_size_1 = subgroup_size;
    subgroup_invocation_id_1 = subgroup_invocation_id;
    main_1();
}

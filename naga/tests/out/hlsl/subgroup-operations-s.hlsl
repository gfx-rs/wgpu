static uint num_subgroups_1 = (uint)0;
static uint subgroup_id_1 = (uint)0;
static uint subgroup_size_1 = (uint)0;
static uint subgroup_invocation_id_1 = (uint)0;

void main_1()
{
    uint _expr5 = subgroup_size_1;
    uint _expr6 = subgroup_invocation_id_1;
    const uint4 _e9 = WaveActiveBallot(((_expr6 & 1u) == 1u));
    const uint4 _e10 = WaveActiveBallot(true);
    const bool _e12 = WaveActiveAllTrue((_expr6 != 0u));
    const bool _e14 = WaveActiveAnyTrue((_expr6 == 0u));
    const uint _e15 = WaveActiveSum(_expr6);
    const uint _e16 = WaveActiveProduct(_expr6);
    const uint _e17 = WaveActiveMin(_expr6);
    const uint _e18 = WaveActiveMax(_expr6);
    const uint _e19 = WaveActiveBitAnd(_expr6);
    const uint _e20 = WaveActiveBitOr(_expr6);
    const uint _e21 = WaveActiveBitXor(_expr6);
    const uint _e22 = WavePrefixSum(_expr6);
    const uint _e23 = WavePrefixProduct(_expr6);
    const uint _e24 = _expr6 + WavePrefixSum(_expr6);
    const uint _e25 = _expr6 * WavePrefixProduct(_expr6);
    const uint _e26 = WaveReadLaneFirst(_expr6);
    const uint _e27 = WaveReadLaneAt(_expr6, 4u);
    const uint _e30 = WaveReadLaneAt(_expr6, ((_expr5 - 1u) - _expr6));
    const uint _e31 = WaveReadLaneAt(_expr6, WaveGetLaneIndex() + 1u);
    const uint _e32 = WaveReadLaneAt(_expr6, WaveGetLaneIndex() - 1u);
    const uint _e34 = WaveReadLaneAt(_expr6, WaveGetLaneIndex() ^ (_expr5 - 1u));
    return;
}

[numthreads(1, 1, 1)]
void main(uint3 __local_invocation_id : SV_GroupThreadID)
{
    if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {
    }
    GroupMemoryBarrierWithGroupSync();
    const uint num_subgroups = (1u + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
    const uint subgroup_id = (__local_invocation_id.x * 1u + __local_invocation_id.y * 1u + __local_invocation_id.z) / WaveGetLaneCount();
    const uint subgroup_size = WaveGetLaneCount();
    const uint subgroup_invocation_id = WaveGetLaneIndex();
    num_subgroups_1 = num_subgroups;
    subgroup_id_1 = subgroup_id;
    subgroup_size_1 = subgroup_size;
    subgroup_invocation_id_1 = subgroup_invocation_id;
    main_1();
}

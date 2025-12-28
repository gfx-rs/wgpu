groupshared uint wg_scalar;
groupshared int wg_signed;

[numthreads(64, 1, 1)]
void test_atomic_workgroup_uniform_load(uint3 workgroup_id : SV_GroupID, uint3 local_id : SV_GroupThreadID, uint3 __local_invocation_id : SV_GroupThreadID)
{
    if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {
        wg_scalar = (uint)0;
        wg_signed = (int)0;
    }
    GroupMemoryBarrierWithGroupSync();
    bool local = (bool)0;

    uint active_tile_index = (workgroup_id.x + (workgroup_id.y * 32768u));
    uint _e11; InterlockedOr(wg_scalar, uint((active_tile_index >= 64u)), _e11);
    int _e14; InterlockedAdd(wg_signed, int(1), _e14);
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    uint _e16 = wg_scalar;
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    int _e18 = wg_signed;
    GroupMemoryBarrierWithGroupSync();
    if ((_e16 == 0u)) {
        local = (_e18 > int(0));
    } else {
        local = false;
    }
    bool _e26 = local;
    if (_e26) {
        return;
    } else {
        return;
    }
}

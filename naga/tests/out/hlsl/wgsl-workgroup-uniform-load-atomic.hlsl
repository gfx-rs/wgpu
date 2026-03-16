struct AtomicStruct {
    uint atomic_scalar;
    int atomic_arr[2];
};

groupshared uint wg_scalar;
groupshared int wg_signed;
groupshared AtomicStruct wg_struct;

[numthreads(64, 1, 1)]
void test_atomic_workgroup_uniform_load(uint3 workgroup_id : SV_GroupID, uint3 local_id : SV_GroupThreadID)
{
    if (all(local_id == uint3(0u, 0u, 0u))) {
        wg_scalar = (uint)0;
        wg_signed = (int)0;
        wg_struct = (AtomicStruct)0;
    }
    GroupMemoryBarrierWithGroupSync();
    bool local = (bool)0;
    bool local_1 = (bool)0;
    bool local_2 = (bool)0;

    uint active_tile_index = (workgroup_id.x + (workgroup_id.y * 32768u));
    uint _e11; InterlockedOr(wg_scalar, uint((active_tile_index >= 64u)), _e11);
    int _e14; InterlockedAdd(wg_signed, int(1), _e14);
    wg_struct.atomic_scalar = 1u;
    int _e22; InterlockedAdd(wg_struct.atomic_arr[0], int(1), _e22);
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    uint _e24 = wg_scalar;
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    int _e26 = wg_signed;
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    uint _e29 = wg_struct.atomic_scalar;
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    int _e33 = wg_struct.atomic_arr[0];
    GroupMemoryBarrierWithGroupSync();
    if ((_e24 == 0u)) {
        local = (_e26 > int(0));
    } else {
        local = false;
    }
    bool _e41 = local;
    if (_e41) {
        local_1 = (_e29 > 0u);
    } else {
        local_1 = false;
    }
    bool _e47 = local_1;
    if (_e47) {
        local_2 = (_e33 > int(0));
    } else {
        local_2 = false;
    }
    bool _e53 = local_2;
    if (_e53) {
        return;
    } else {
        return;
    }
}

static const uint SIZE = 128u;

groupshared int arr_i32_[128];

[numthreads(4, 1, 1)]
void test_workgroupUniformLoad(uint3 workgroup_id : SV_GroupID, uint local_invocation_index : SV_GroupIndex)
{
    uint zero_init_index = (uint)0;

    zero_init_index = local_invocation_index;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e11 = zero_init_index;
        if ((_e11 >= 128u)) {
            break;
        }
        arr_i32_[min(uint(_e11), 127u)] = int(0);
        zero_init_index = (_e11 + 4u);
    }
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    int _e4 = arr_i32_[min(uint(workgroup_id.x), 127u)];
    GroupMemoryBarrierWithGroupSync();
    if ((_e4 > int(10))) {
        GroupMemoryBarrierWithGroupSync();
        return;
    } else {
        return;
    }
}

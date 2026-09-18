static const uint SIZE = 128u;

groupshared int arr_i32_[128];

[numthreads(4, 1, 1)]
void test_workgroupUniformLoad(uint3 workgroup_id : SV_GroupID, uint local_invocation_index : SV_GroupIndex)
{
    [loop]
    for (uint zero_flat_index = local_invocation_index; zero_flat_index < 128u; zero_flat_index += 4u) {
        uint zero_index = (zero_flat_index / 1u) % 128u;
        arr_i32_[zero_index] = (int)0;
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

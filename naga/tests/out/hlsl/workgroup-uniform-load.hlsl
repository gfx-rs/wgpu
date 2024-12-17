static const uint SIZE = 128u;

groupshared int arr_i32_[128];

[numthreads(4, 1, 1)]
void test_workgroupUniformLoad(uint3 workgroup_id : SV_GroupID, uint3 __local_invocation_id : SV_GroupThreadID)
{
    if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {
        arr_i32_ = (int[128])0;
    }
    GroupMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    int _e4 = arr_i32_[min(uint(workgroup_id.x), 127u)];
    GroupMemoryBarrierWithGroupSync();
    if ((_e4 > 10)) {
        GroupMemoryBarrierWithGroupSync();
        return;
    } else {
        return;
    }
}

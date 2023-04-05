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
    int _expr4 = arr_i32_[workgroup_id.x];
    GroupMemoryBarrierWithGroupSync();
    if ((_expr4 > 10)) {
        GroupMemoryBarrierWithGroupSync();
        return;
    } else {
        return;
    }
}

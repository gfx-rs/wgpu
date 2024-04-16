static const uint SIZE = 128u;

groupshared int arr_i32_[128];

[numthreads(4, 1, 1)]
void test_workgroupUniformLoad(uint3 workgroup_id : SV_GroupID, uint __local_invocation_index : SV_GroupIndex)
{
    arr_i32_[__local_invocation_index] = (int)0;
    arr_i32_[__local_invocation_index + 4u] = (int)0;
    arr_i32_[__local_invocation_index + 8u] = (int)0;
    arr_i32_[__local_invocation_index + 12u] = (int)0;
    arr_i32_[__local_invocation_index + 16u] = (int)0;
    arr_i32_[__local_invocation_index + 20u] = (int)0;
    arr_i32_[__local_invocation_index + 24u] = (int)0;
    arr_i32_[__local_invocation_index + 28u] = (int)0;
    arr_i32_[__local_invocation_index + 32u] = (int)0;
    arr_i32_[__local_invocation_index + 36u] = (int)0;
    arr_i32_[__local_invocation_index + 40u] = (int)0;
    arr_i32_[__local_invocation_index + 44u] = (int)0;
    arr_i32_[__local_invocation_index + 48u] = (int)0;
    arr_i32_[__local_invocation_index + 52u] = (int)0;
    arr_i32_[__local_invocation_index + 56u] = (int)0;
    arr_i32_[__local_invocation_index + 60u] = (int)0;
    arr_i32_[__local_invocation_index + 64u] = (int)0;
    arr_i32_[__local_invocation_index + 68u] = (int)0;
    arr_i32_[__local_invocation_index + 72u] = (int)0;
    arr_i32_[__local_invocation_index + 76u] = (int)0;
    arr_i32_[__local_invocation_index + 80u] = (int)0;
    arr_i32_[__local_invocation_index + 84u] = (int)0;
    arr_i32_[__local_invocation_index + 88u] = (int)0;
    arr_i32_[__local_invocation_index + 92u] = (int)0;
    arr_i32_[__local_invocation_index + 96u] = (int)0;
    arr_i32_[__local_invocation_index + 100u] = (int)0;
    arr_i32_[__local_invocation_index + 104u] = (int)0;
    arr_i32_[__local_invocation_index + 108u] = (int)0;
    arr_i32_[__local_invocation_index + 112u] = (int)0;
    arr_i32_[__local_invocation_index + 116u] = (int)0;
    arr_i32_[__local_invocation_index + 120u] = (int)0;
    arr_i32_[__local_invocation_index + 124u] = (int)0;
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

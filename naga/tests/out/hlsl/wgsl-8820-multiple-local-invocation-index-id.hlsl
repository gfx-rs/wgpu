struct Input {
    uint3 local_invocation_id : SV_GroupThreadID;
    uint local_invocation_index : SV_GroupIndex;
};

groupshared uint wg_var;

[numthreads(1, 1, 1)]
void compute1_(Input input, uint local_invocation_index : SV_GroupIndex)
{
    if (local_invocation_index == 0) {
        wg_var = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    wg_var = (input.local_invocation_index * 2u);
    uint _e6 = wg_var;
    wg_var = (_e6 + input.local_invocation_id.x);
    return;
}

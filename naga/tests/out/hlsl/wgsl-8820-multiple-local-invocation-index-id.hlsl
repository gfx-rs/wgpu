struct Input {
    uint3 local_invocation_id : SV_GroupThreadID;
    uint local_invocation_index : SV_GroupIndex;
};

groupshared uint wg_var;

[numthreads(1, 1, 1)]
void compute1_(Input input)
{
    if ((input.local_invocation_index == 0u)) {
        wg_var = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    wg_var = (input.local_invocation_index * 2u);
    uint _e6 = wg_var;
    wg_var = (_e6 + input.local_invocation_id.x);
    return;
}

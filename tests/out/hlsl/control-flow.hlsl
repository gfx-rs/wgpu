struct ComputeInput_main {
    uint3 global_id1 : SV_DispatchThreadID;
};

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    return;
}

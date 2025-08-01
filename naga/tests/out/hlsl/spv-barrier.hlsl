void function()
{
    GroupMemoryBarrier();
    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrier();
    DeviceMemoryBarrier();
    DeviceMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();
    DeviceMemoryBarrier();
    GroupMemoryBarrier();
    DeviceMemoryBarrier();
    DeviceMemoryBarrierWithGroupSync();
    GroupMemoryBarrierWithGroupSync();
    DeviceMemoryBarrierWithGroupSync();
    return;
}

[numthreads(64, 1, 1)]
void main()
{
    function();
}

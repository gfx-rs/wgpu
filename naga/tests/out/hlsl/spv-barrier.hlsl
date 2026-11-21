[numthreads(64, 1, 1)]
void main()
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
}

RWByteAddressBuffer payload_ : register(u0);
RWByteAddressBuffer flag : register(u1);
groupshared uint stage;

[numthreads(64, 1, 1)]
void main(uint index : SV_GroupIndex)
{
    if (index == 0) {
        stage = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint spins = 0u;

    payload_.Store(index*4+0, asuint(index));
    DeviceMemoryBarrier();
    if ((index == 0u)) {
        { uint dummy = 0; flag.InterlockedExchange(0, 1u, dummy); }
    }
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e11; flag.InterlockedOr(0, 0, _e11);
        if ((_e11 != 0u)) {
            DeviceMemoryBarrier();
            uint _e18 = asuint(payload_.Load(0+0));
            stage = _e18;
            GroupMemoryBarrier();
            break;
        }
        uint _e19 = spins;
        spins = (_e19 + 1u);
        uint _e22 = spins;
        if ((_e22 > 65536u)) {
            break;
        }
    }
    return;
}

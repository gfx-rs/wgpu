struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct Struct {
    uint64_t atomic_scalar;
    uint64_t atomic_arr[2];
};

RWByteAddressBuffer storage_atomic_scalar : register(u0);
RWByteAddressBuffer storage_atomic_arr : register(u1);
RWByteAddressBuffer storage_struct : register(u2);
cbuffer input : register(b3) { uint64_t input; }

[numthreads(2, 1, 1)]
void cs_main(uint3 id : SV_GroupThreadID)
{
    uint64_t _e3 = input;
    storage_atomic_scalar.InterlockedMax64(0, _e3);
    uint64_t _e7 = input;
    storage_atomic_arr.InterlockedMax64(8, (1uL + _e7));
    storage_struct.InterlockedMax64(0, 1uL);
    storage_struct.InterlockedMax64(8+8, uint64_t(id.x));
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e20 = input;
    storage_atomic_scalar.InterlockedMin64(0, _e20);
    uint64_t _e24 = input;
    storage_atomic_arr.InterlockedMin64(8, (1uL + _e24));
    storage_struct.InterlockedMin64(0, 1uL);
    storage_struct.InterlockedMin64(8+8, uint64_t(id.x));
    return;
}

struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct Struct {
    uint64_t atomic_scalar;
    int64_t atomic_arr[2];
};

RWByteAddressBuffer storage_atomic_scalar : register(u0);
RWByteAddressBuffer storage_atomic_arr : register(u1);
RWByteAddressBuffer storage_struct : register(u2);
groupshared uint64_t workgroup_atomic_scalar;
groupshared int64_t workgroup_atomic_arr[2];
groupshared Struct workgroup_struct;

[numthreads(2, 1, 1)]
void cs_main(uint3 id : SV_GroupThreadID, uint3 __local_invocation_id : SV_GroupThreadID)
{
    if (all(__local_invocation_id == uint3(0u, 0u, 0u))) {
        workgroup_atomic_scalar = (uint64_t)0;
        workgroup_atomic_arr = (int64_t[2])0;
        workgroup_struct = (Struct)0;
    }
    GroupMemoryBarrierWithGroupSync();
    storage_atomic_scalar.Store(0, 1uL);
    storage_atomic_arr.Store(8, 1L);
    storage_struct.Store(0, 1uL);
    storage_struct.Store(8+8, 1L);
    workgroup_atomic_scalar = 1uL;
    workgroup_atomic_arr[1] = 1L;
    workgroup_struct.atomic_scalar = 1uL;
    workgroup_struct.atomic_arr[1] = 1L;
    GroupMemoryBarrierWithGroupSync();
    uint64_t l0_ = storage_atomic_scalar.Load<uint64_t>(0);
    int64_t l1_ = storage_atomic_arr.Load<int64_t>(8);
    uint64_t l2_ = storage_struct.Load<uint64_t>(0);
    int64_t l3_ = storage_struct.Load<int64_t>(8+8);
    uint64_t l4_ = workgroup_atomic_scalar;
    int64_t l5_ = workgroup_atomic_arr[1];
    uint64_t l6_ = workgroup_struct.atomic_scalar;
    int64_t l7_ = workgroup_struct.atomic_arr[1];
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e51; storage_atomic_scalar.InterlockedAdd64(0, 1uL, _e51);
    int64_t _e55; storage_atomic_arr.InterlockedAdd64(8, 1L, _e55);
    uint64_t _e59; storage_struct.InterlockedAdd64(0, 1uL, _e59);
    int64_t _e64; storage_struct.InterlockedAdd64(8+8, 1L, _e64);
    uint64_t _e67; InterlockedAdd(workgroup_atomic_scalar, 1uL, _e67);
    int64_t _e71; InterlockedAdd(workgroup_atomic_arr[1], 1L, _e71);
    uint64_t _e75; InterlockedAdd(workgroup_struct.atomic_scalar, 1uL, _e75);
    int64_t _e80; InterlockedAdd(workgroup_struct.atomic_arr[1], 1L, _e80);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e83; storage_atomic_scalar.InterlockedAdd64(0, -1uL, _e83);
    int64_t _e87; storage_atomic_arr.InterlockedAdd64(8, -1L, _e87);
    uint64_t _e91; storage_struct.InterlockedAdd64(0, -1uL, _e91);
    int64_t _e96; storage_struct.InterlockedAdd64(8+8, -1L, _e96);
    uint64_t _e99; InterlockedAdd(workgroup_atomic_scalar, -1uL, _e99);
    int64_t _e103; InterlockedAdd(workgroup_atomic_arr[1], -1L, _e103);
    uint64_t _e107; InterlockedAdd(workgroup_struct.atomic_scalar, -1uL, _e107);
    int64_t _e112; InterlockedAdd(workgroup_struct.atomic_arr[1], -1L, _e112);
    GroupMemoryBarrierWithGroupSync();
    storage_atomic_scalar.InterlockedMax64(0, 1uL);
    storage_atomic_arr.InterlockedMax64(8, 1L);
    storage_struct.InterlockedMax64(0, 1uL);
    storage_struct.InterlockedMax64(8+8, 1L);
    InterlockedMax(workgroup_atomic_scalar, 1uL);
    InterlockedMax(workgroup_atomic_arr[1], 1L);
    InterlockedMax(workgroup_struct.atomic_scalar, 1uL);
    InterlockedMax(workgroup_struct.atomic_arr[1], 1L);
    GroupMemoryBarrierWithGroupSync();
    storage_atomic_scalar.InterlockedMin64(0, 1uL);
    storage_atomic_arr.InterlockedMin64(8, 1L);
    storage_struct.InterlockedMin64(0, 1uL);
    storage_struct.InterlockedMin64(8+8, 1L);
    InterlockedMin(workgroup_atomic_scalar, 1uL);
    InterlockedMin(workgroup_atomic_arr[1], 1L);
    InterlockedMin(workgroup_struct.atomic_scalar, 1uL);
    InterlockedMin(workgroup_struct.atomic_arr[1], 1L);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e163; storage_atomic_scalar.InterlockedAnd64(0, 1uL, _e163);
    int64_t _e167; storage_atomic_arr.InterlockedAnd64(8, 1L, _e167);
    uint64_t _e171; storage_struct.InterlockedAnd64(0, 1uL, _e171);
    int64_t _e176; storage_struct.InterlockedAnd64(8+8, 1L, _e176);
    uint64_t _e179; InterlockedAnd(workgroup_atomic_scalar, 1uL, _e179);
    int64_t _e183; InterlockedAnd(workgroup_atomic_arr[1], 1L, _e183);
    uint64_t _e187; InterlockedAnd(workgroup_struct.atomic_scalar, 1uL, _e187);
    int64_t _e192; InterlockedAnd(workgroup_struct.atomic_arr[1], 1L, _e192);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e195; storage_atomic_scalar.InterlockedOr64(0, 1uL, _e195);
    int64_t _e199; storage_atomic_arr.InterlockedOr64(8, 1L, _e199);
    uint64_t _e203; storage_struct.InterlockedOr64(0, 1uL, _e203);
    int64_t _e208; storage_struct.InterlockedOr64(8+8, 1L, _e208);
    uint64_t _e211; InterlockedOr(workgroup_atomic_scalar, 1uL, _e211);
    int64_t _e215; InterlockedOr(workgroup_atomic_arr[1], 1L, _e215);
    uint64_t _e219; InterlockedOr(workgroup_struct.atomic_scalar, 1uL, _e219);
    int64_t _e224; InterlockedOr(workgroup_struct.atomic_arr[1], 1L, _e224);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e227; storage_atomic_scalar.InterlockedXor64(0, 1uL, _e227);
    int64_t _e231; storage_atomic_arr.InterlockedXor64(8, 1L, _e231);
    uint64_t _e235; storage_struct.InterlockedXor64(0, 1uL, _e235);
    int64_t _e240; storage_struct.InterlockedXor64(8+8, 1L, _e240);
    uint64_t _e243; InterlockedXor(workgroup_atomic_scalar, 1uL, _e243);
    int64_t _e247; InterlockedXor(workgroup_atomic_arr[1], 1L, _e247);
    uint64_t _e251; InterlockedXor(workgroup_struct.atomic_scalar, 1uL, _e251);
    int64_t _e256; InterlockedXor(workgroup_struct.atomic_arr[1], 1L, _e256);
    uint64_t _e259; storage_atomic_scalar.InterlockedExchange64(0, 1uL, _e259);
    int64_t _e263; storage_atomic_arr.InterlockedExchange64(8, 1L, _e263);
    uint64_t _e267; storage_struct.InterlockedExchange64(0, 1uL, _e267);
    int64_t _e272; storage_struct.InterlockedExchange64(8+8, 1L, _e272);
    uint64_t _e275; InterlockedExchange(workgroup_atomic_scalar, 1uL, _e275);
    int64_t _e279; InterlockedExchange(workgroup_atomic_arr[1], 1L, _e279);
    uint64_t _e283; InterlockedExchange(workgroup_struct.atomic_scalar, 1uL, _e283);
    int64_t _e288; InterlockedExchange(workgroup_struct.atomic_arr[1], 1L, _e288);
    return;
}

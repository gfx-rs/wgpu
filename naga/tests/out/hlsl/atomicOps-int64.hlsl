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
    uint64_t _e51; storage_atomic_scalar.InterlockedAdd(0, 1uL, _e51);
    int64_t _e55; storage_atomic_arr.InterlockedAdd(8, 1L, _e55);
    uint64_t _e59; storage_struct.InterlockedAdd(0, 1uL, _e59);
    int64_t _e64; storage_struct.InterlockedAdd(8+8, 1L, _e64);
    uint64_t _e67; InterlockedAdd(workgroup_atomic_scalar, 1uL, _e67);
    int64_t _e71; InterlockedAdd(workgroup_atomic_arr[1], 1L, _e71);
    uint64_t _e75; InterlockedAdd(workgroup_struct.atomic_scalar, 1uL, _e75);
    int64_t _e80; InterlockedAdd(workgroup_struct.atomic_arr[1], 1L, _e80);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e83; storage_atomic_scalar.InterlockedAdd(0, -1uL, _e83);
    int64_t _e87; storage_atomic_arr.InterlockedAdd(8, -1L, _e87);
    uint64_t _e91; storage_struct.InterlockedAdd(0, -1uL, _e91);
    int64_t _e96; storage_struct.InterlockedAdd(8+8, -1L, _e96);
    uint64_t _e99; InterlockedAdd(workgroup_atomic_scalar, -1uL, _e99);
    int64_t _e103; InterlockedAdd(workgroup_atomic_arr[1], -1L, _e103);
    uint64_t _e107; InterlockedAdd(workgroup_struct.atomic_scalar, -1uL, _e107);
    int64_t _e112; InterlockedAdd(workgroup_struct.atomic_arr[1], -1L, _e112);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e115; storage_atomic_scalar.InterlockedMax(0, 1uL, _e115);
    int64_t _e119; storage_atomic_arr.InterlockedMax(8, 1L, _e119);
    uint64_t _e123; storage_struct.InterlockedMax(0, 1uL, _e123);
    int64_t _e128; storage_struct.InterlockedMax(8+8, 1L, _e128);
    uint64_t _e131; InterlockedMax(workgroup_atomic_scalar, 1uL, _e131);
    int64_t _e135; InterlockedMax(workgroup_atomic_arr[1], 1L, _e135);
    uint64_t _e139; InterlockedMax(workgroup_struct.atomic_scalar, 1uL, _e139);
    int64_t _e144; InterlockedMax(workgroup_struct.atomic_arr[1], 1L, _e144);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e147; storage_atomic_scalar.InterlockedMin(0, 1uL, _e147);
    int64_t _e151; storage_atomic_arr.InterlockedMin(8, 1L, _e151);
    uint64_t _e155; storage_struct.InterlockedMin(0, 1uL, _e155);
    int64_t _e160; storage_struct.InterlockedMin(8+8, 1L, _e160);
    uint64_t _e163; InterlockedMin(workgroup_atomic_scalar, 1uL, _e163);
    int64_t _e167; InterlockedMin(workgroup_atomic_arr[1], 1L, _e167);
    uint64_t _e171; InterlockedMin(workgroup_struct.atomic_scalar, 1uL, _e171);
    int64_t _e176; InterlockedMin(workgroup_struct.atomic_arr[1], 1L, _e176);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e179; storage_atomic_scalar.InterlockedAnd(0, 1uL, _e179);
    int64_t _e183; storage_atomic_arr.InterlockedAnd(8, 1L, _e183);
    uint64_t _e187; storage_struct.InterlockedAnd(0, 1uL, _e187);
    int64_t _e192; storage_struct.InterlockedAnd(8+8, 1L, _e192);
    uint64_t _e195; InterlockedAnd(workgroup_atomic_scalar, 1uL, _e195);
    int64_t _e199; InterlockedAnd(workgroup_atomic_arr[1], 1L, _e199);
    uint64_t _e203; InterlockedAnd(workgroup_struct.atomic_scalar, 1uL, _e203);
    int64_t _e208; InterlockedAnd(workgroup_struct.atomic_arr[1], 1L, _e208);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e211; storage_atomic_scalar.InterlockedOr(0, 1uL, _e211);
    int64_t _e215; storage_atomic_arr.InterlockedOr(8, 1L, _e215);
    uint64_t _e219; storage_struct.InterlockedOr(0, 1uL, _e219);
    int64_t _e224; storage_struct.InterlockedOr(8+8, 1L, _e224);
    uint64_t _e227; InterlockedOr(workgroup_atomic_scalar, 1uL, _e227);
    int64_t _e231; InterlockedOr(workgroup_atomic_arr[1], 1L, _e231);
    uint64_t _e235; InterlockedOr(workgroup_struct.atomic_scalar, 1uL, _e235);
    int64_t _e240; InterlockedOr(workgroup_struct.atomic_arr[1], 1L, _e240);
    GroupMemoryBarrierWithGroupSync();
    uint64_t _e243; storage_atomic_scalar.InterlockedXor(0, 1uL, _e243);
    int64_t _e247; storage_atomic_arr.InterlockedXor(8, 1L, _e247);
    uint64_t _e251; storage_struct.InterlockedXor(0, 1uL, _e251);
    int64_t _e256; storage_struct.InterlockedXor(8+8, 1L, _e256);
    uint64_t _e259; InterlockedXor(workgroup_atomic_scalar, 1uL, _e259);
    int64_t _e263; InterlockedXor(workgroup_atomic_arr[1], 1L, _e263);
    uint64_t _e267; InterlockedXor(workgroup_struct.atomic_scalar, 1uL, _e267);
    int64_t _e272; InterlockedXor(workgroup_struct.atomic_arr[1], 1L, _e272);
    uint64_t _e275; storage_atomic_scalar.InterlockedExchange(0, 1uL, _e275);
    int64_t _e279; storage_atomic_arr.InterlockedExchange(8, 1L, _e279);
    uint64_t _e283; storage_struct.InterlockedExchange(0, 1uL, _e283);
    int64_t _e288; storage_struct.InterlockedExchange(8+8, 1L, _e288);
    uint64_t _e291; InterlockedExchange(workgroup_atomic_scalar, 1uL, _e291);
    int64_t _e295; InterlockedExchange(workgroup_atomic_arr[1], 1L, _e295);
    uint64_t _e299; InterlockedExchange(workgroup_struct.atomic_scalar, 1uL, _e299);
    int64_t _e304; InterlockedExchange(workgroup_struct.atomic_arr[1], 1L, _e304);
    return;
}

struct Padded {
    uint3 xyz;
    uint tail;
};

static const uint W = 64u;
static const uint N = 513u;

groupshared uint flag;
groupshared uint2 pair;
groupshared uint scratch[513];
groupshared uint counter;
groupshared uint3 vectors[3];
groupshared Padded padded;
RWByteAddressBuffer output : register(u0);

uint sum_small()
{
    uint sum_1 = (uint)0;
    uint j_3 = 0u;

    uint _e1 = flag;
    uint _e4 = pair.x;
    uint _e8 = pair.y;
    uint _e13 = padded.xyz.x;
    uint _e18 = padded.xyz.y;
    uint _e23 = padded.xyz.z;
    uint _e27 = padded.tail;
    sum_1 = ((((((_e1 + _e4) + _e8) + _e13) + _e18) + _e23) + _e27);
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e55 = j_3;
            j_3 = (_e55 + 1u);
        }
        loop_init = false;
        uint _e32 = j_3;
        if ((_e32 < 3u)) {
        } else {
            break;
        }
        {
            uint _e35 = sum_1;
            uint _e37 = j_3;
            uint _e40 = vectors[min(uint(_e37), 2u)].x;
            uint _e42 = j_3;
            uint _e45 = vectors[min(uint(_e42), 2u)].y;
            uint _e48 = j_3;
            uint _e51 = vectors[min(uint(_e48), 2u)].z;
            sum_1 = (_e35 + ((_e40 + _e45) + _e51));
        }
    }
    uint _e57 = sum_1;
    return _e57;
}

typedef uint3 ret_Constructarray3_uint3_[3];
ret_Constructarray3_uint3_ Constructarray3_uint3_(uint3 arg0, uint3 arg1, uint3 arg2) {
    uint3 ret[3] = { arg0, arg1, arg2 };
    return ret;
}

Padded ConstructPadded(uint3 arg0, uint arg1) {
    Padded ret = (Padded)0;
    ret.xyz = arg0;
    ret.tail = arg1;
    return ret;
}

[numthreads(64, 1, 1)]
void main(uint i : SV_GroupIndex, uint3 group : SV_GroupID)
{
    if (i == 0) {
        flag = (uint)0;
    }
    if (i == 0) {
        pair = (uint2)0;
    }
    [loop]
    for (uint zero_flat_index = i; zero_flat_index < 513u; zero_flat_index += 64u) {
        uint zero_index = zero_flat_index;
        scratch[zero_index] = (uint)0;
    }
    if (i == 0) {
        counter = (uint)0;
    }
    [loop]
    for (uint zero_flat_index_1 = i; zero_flat_index_1 < 3u; zero_flat_index_1 += 64u) {
        uint zero_index_1 = zero_flat_index_1;
        vectors[zero_index_1] = (uint3)0;
    }
    if (i == 0) {
        padded.xyz = (uint3)0;
    }
    if (i == 0) {
        padded.tail = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint initial = (uint)0;
    uint j = 0u;
    uint j_1 = (uint)0;
    uint sum = 0u;
    uint j_2 = 0u;

    uint base = (group.x * 4u);
    if ((i == 0u)) {
        const uint _e7 = sum_small();
        uint _e9; InterlockedOr(counter, 0, _e9);
        initial = (_e7 + _e9);
        uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
        bool loop_init_1 = true;
        while(true) {
            if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
            loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
            if (!loop_init_1) {
                uint _e24 = j;
                j = (_e24 + 1u);
            }
            loop_init_1 = false;
            uint _e14 = j;
            if ((_e14 < N)) {
            } else {
                break;
            }
            {
                uint _e17 = initial;
                uint _e19 = j;
                uint _e21 = scratch[min(uint(_e19), 512u)];
                initial = (_e17 + _e21);
            }
        }
        uint _e28 = initial;
        output.Store(base*4, asuint(_e28));
    }
    GroupMemoryBarrierWithGroupSync();
    j_1 = i;
    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e39 = j_1;
            j_1 = (_e39 + W);
        }
        loop_init_2 = false;
        uint _e30 = j_1;
        if ((_e30 < N)) {
        } else {
            break;
        }
        {
            uint _e34 = j_1;
            uint _e36 = j_1;
            scratch[min(uint(_e34), 512u)] = (_e36 + group.x);
        }
    }
    uint _e44; InterlockedAdd(counter, 1u, _e44);
    if ((i == 0u)) {
        flag = 1u;
        pair = uint2(2u, 3u);
        vectors = Constructarray3_uint3_(uint3(4u, 5u, 6u), uint3(7u, 8u, 9u), uint3(10u, 11u, 12u));
        padded = ConstructPadded(uint3(13u, 14u, 15u), 16u);
    }
    GroupMemoryBarrierWithGroupSync();
    if ((i == 0u)) {
        uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
        bool loop_init_3 = true;
        while(true) {
            if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
            loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
            if (!loop_init_3) {
                uint _e90 = j_2;
                j_2 = (_e90 + 1u);
            }
            loop_init_3 = false;
            uint _e80 = j_2;
            if ((_e80 < N)) {
            } else {
                break;
            }
            {
                uint _e83 = sum;
                uint _e85 = j_2;
                uint _e87 = scratch[min(uint(_e85), 512u)];
                sum = (_e83 + _e87);
            }
        }
        uint _e96 = sum;
        output.Store((base + 1u)*4, asuint(_e96));
        uint _e102; InterlockedOr(counter, 0, _e102);
        output.Store((base + 2u)*4, asuint(_e102));
        const uint _e107 = sum_small();
        output.Store((base + 3u)*4, asuint(_e107));
        return;
    } else {
        return;
    }
}

[numthreads(1, 1, 1)]
void padded_only(uint3 group_1 : SV_GroupID, uint local_invocation_index : SV_GroupIndex)
{
    if (local_invocation_index == 0) {
        padded.xyz = (uint3)0;
    }
    if (local_invocation_index == 0) {
        padded.tail = (uint)0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint base_1 = (group_1.x * 4u);
    uint _e9 = padded.xyz.x;
    uint _e13 = padded.xyz.y;
    uint _e18 = padded.xyz.z;
    uint _e22 = padded.tail;
    output.Store(base_1*4, asuint((((_e9 + _e13) + _e18) + _e22)));
    padded = ConstructPadded(uint3(13u, 14u, 15u), 16u);
    uint _e38 = padded.xyz.x;
    uint _e42 = padded.xyz.y;
    uint _e47 = padded.xyz.z;
    uint _e51 = padded.tail;
    output.Store((base_1 + 1u)*4, asuint((((_e38 + _e42) + _e47) + _e51)));
    output.Store((base_1 + 2u)*4, asuint(0u));
    output.Store((base_1 + 3u)*4, asuint(0u));
    return;
}

[numthreads(1, 1, 1)]
void no_workgroup(uint3 group_2 : SV_GroupID)
{
    uint base_2 = (group_2.x * 4u);
    output.Store(base_2*4, asuint(group_2.x));
    output.Store((base_2 + 1u)*4, asuint(0u));
    output.Store((base_2 + 2u)*4, asuint(0u));
    output.Store((base_2 + 3u)*4, asuint(0u));
    return;
}

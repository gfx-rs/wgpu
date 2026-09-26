struct Particle {
    float4 pos;
    float4 vel;
};

struct WithAtomics {
    uint count;
    uint bins[100];
};

struct Mixed {
    row_major float4x4 transform;
    uint flag;
    int _pad2_0;
    int _pad2_1;
    int _pad2_2;
    float4 grid[18][18];
    WithAtomics cells[3];
    int _end_pad_0;
};

struct Inputs {
    uint3 group : SV_GroupID;
    uint index : SV_GroupIndex;
};

groupshared uint scalar;
groupshared float small[16];
groupshared Particle particles[256];
groupshared uint2 single[1];
groupshared Mixed mixed;
groupshared int counter;
RWByteAddressBuffer output : register(u0);

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

uint ZeroValueuint() {
    return (uint)0;
}

float ZeroValuefloat() {
    return (float)0;
}

Particle ZeroValueParticle() {
    return (Particle)0;
}

uint2 ZeroValueuint2() {
    return (uint2)0;
}

float4x4 ZeroValuefloat4x4() {
    return (float4x4)0;
}

float4 ZeroValuefloat4() {
    return (float4)0;
}

int ZeroValueint() {
    return (int)0;
}

[numthreads(8, 4, 1)]
void main(uint3 id : SV_GroupThreadID, uint local_invocation_index : SV_GroupIndex)
{
    uint zero_init_index = (uint)0;

    if ((local_invocation_index < 16u)) {
        small[min(uint(local_invocation_index), 15u)] = ZeroValuefloat();
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            uint _e92 = zero_init_index;
            uint _e93 = (_e92 + 32u);
            zero_init_index = _e93;
            if ((_e93 >= 256u)) {
                break;
            }
        }
        loop_init = false;
        uint _e86 = zero_init_index;
        particles[min(uint(_e86), 255u)] = ZeroValueParticle();
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound_1 = uint2(4294967295u, 4294967295u);
    bool loop_init_1 = true;
    while(true) {
        if (all(loop_bound_1 == uint2(0u, 0u))) { break; }
        loop_bound_1 -= uint2(loop_bound_1.y == 0u, 1u);
        if (!loop_init_1) {
            uint _e121 = zero_init_index;
            uint _e122 = (_e121 + 32u);
            zero_init_index = _e122;
            if ((_e122 >= 324u)) {
                break;
            }
        }
        loop_init_1 = false;
        uint _e110 = zero_init_index;
        mixed.grid[min(uint(naga_div(_e110, 18u)), 17u)][min(uint(naga_mod(_e110, 18u)), 17u)] = ZeroValuefloat4();
    }
    if ((local_invocation_index < 3u)) {
        { uint dummy = 0; InterlockedExchange(mixed.cells[min(uint(local_invocation_index), 2u)].count, ZeroValueuint(), dummy); }
    }
    zero_init_index = local_invocation_index;
    uint2 loop_bound_2 = uint2(4294967295u, 4294967295u);
    bool loop_init_2 = true;
    while(true) {
        if (all(loop_bound_2 == uint2(0u, 0u))) { break; }
        loop_bound_2 -= uint2(loop_bound_2.y == 0u, 1u);
        if (!loop_init_2) {
            uint _e146 = zero_init_index;
            uint _e147 = (_e146 + 32u);
            zero_init_index = _e147;
            if ((_e147 >= 300u)) {
                break;
            }
        }
        loop_init_2 = false;
        uint _e134 = zero_init_index;
        { uint dummy_1 = 0; InterlockedExchange(mixed.cells[min(uint(naga_div(_e134, 100u)), 2u)].bins[min(uint(naga_mod(_e134, 100u)), 99u)], ZeroValueuint(), dummy_1); }
    }
    if ((local_invocation_index == 0u)) {
        scalar = ZeroValueuint();
        single[0u] = ZeroValueuint2();
        mixed.transform = ZeroValuefloat4x4();
        mixed.flag = ZeroValueuint();
        { int dummy_2 = 0; InterlockedExchange(counter, ZeroValueint(), dummy_2); }
    }
    GroupMemoryBarrierWithGroupSync();
    uint base = ((id.x + (id.y * 8u)) * 4u);
    uint _e11 = scalar;
    float _e15 = small[3];
    float _e21 = particles[7].vel.y;
    uint _e26 = single[0].x;
    output.Store(base*4, asuint((((float(_e11) + _e15) + _e21) + float(_e26))));
    float _e37 = mixed.transform[1].x;
    uint _e40 = mixed.flag;
    float _e48 = mixed.grid[2][5].z;
    output.Store((base + 1u)*4, asuint(((_e37 + float(_e40)) + _e48)));
    uint _e59; InterlockedOr(mixed.cells[1].bins[42], 0, _e59);
    uint _e64; InterlockedOr(mixed.cells[2].count, 0, _e64);
    output.Store((base + 2u)*4, asuint(float((_e59 + _e64))));
    int _e72; InterlockedOr(counter, 0, _e72);
    output.Store((base + 3u)*4, asuint(float(_e72)));
    return;
}

[numthreads(64, 1, 1)]
void with_index(uint index : SV_GroupIndex)
{
    if ((index < 16u)) {
        small[min(uint(index), 15u)] = ZeroValuefloat();
    }
    GroupMemoryBarrierWithGroupSync();
    float _e7 = small[min(uint(naga_mod(index, 16u)), 15u)];
    output.Store(index*4, asuint(_e7));
    return;
}

[numthreads(16, 1, 1)]
void with_struct(Inputs inputs)
{
    uint zero_init_index_1 = (uint)0;

    zero_init_index_1 = inputs.index;
    uint2 loop_bound_3 = uint2(4294967295u, 4294967295u);
    bool loop_init_3 = true;
    while(true) {
        if (all(loop_bound_3 == uint2(0u, 0u))) { break; }
        loop_bound_3 -= uint2(loop_bound_3.y == 0u, 1u);
        if (!loop_init_3) {
            uint _e24 = zero_init_index_1;
            uint _e25 = (_e24 + 16u);
            zero_init_index_1 = _e25;
            if ((_e25 >= 256u)) {
                break;
            }
        }
        loop_init_3 = false;
        uint _e18 = zero_init_index_1;
        particles[min(uint(_e18), 255u)] = ZeroValueParticle();
    }
    if ((inputs.index == 0u)) {
        { int dummy_3 = 0; InterlockedExchange(counter, ZeroValueint(), dummy_3); }
    }
    GroupMemoryBarrierWithGroupSync();
    float _e9 = particles[min(uint(inputs.index), 255u)].pos.x;
    int _e11; InterlockedOr(counter, 0, _e11);
    output.Store(inputs.index*4, asuint((_e9 + float(_e11))));
    return;
}

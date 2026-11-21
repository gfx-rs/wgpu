struct Particle {
    float3 position;
    uint hits;
    float history[4];
};

groupshared uint total;
groupshared uint small[3];
groupshared float4 exact[8];
groupshared float grid[5][4];
groupshared Particle particles[2];

uint naga_mod(uint lhs, uint rhs) {
    return lhs % (rhs == 0u ? 1u : rhs);
}

uint naga_div(uint lhs, uint rhs) {
    return lhs / (rhs == 0u ? 1u : rhs);
}

float4 ZeroValuefloat4() {
    return (float4)0;
}

float3 ZeroValuefloat3() {
    return (float3)0;
}

uint naga_f2u32(float value) {
    return uint(clamp(value, 0.0, 4294967000.0));
}

[numthreads(4, 2, 1)]
void main(uint index : SV_GroupIndex)
{
    uint zero_init_index = (uint)0;

    if ((index < 3u)) {
        small[min(uint(index), 2u)] = 0u;
    }
    exact[min(uint(index), 7u)] = ZeroValuefloat4();
    zero_init_index = index;
    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        uint _e63 = zero_init_index;
        if ((_e63 >= 20u)) {
            break;
        }
        grid[min(uint(naga_div(_e63, 4u)), 4u)][min(uint(naga_mod(_e63, 4u)), 3u)] = 0.0;
        zero_init_index = (_e63 + 8u);
    }
    if ((index < 2u)) {
        particles[min(uint(index), 1u)].position = ZeroValuefloat3();
        { uint dummy = 0; InterlockedExchange(particles[min(uint(index), 1u)].hits, 0u, dummy); }
    }
    particles[min(uint(naga_div(index, 4u)), 1u)].history[min(uint(naga_mod(index, 4u)), 3u)] = 0.0;
    if ((index == 0u)) {
        total = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint _e5 = small[min(uint(naga_mod(index, 3u)), 2u)];
    float _e10 = exact[min(uint(index), 7u)].x;
    float _e19 = grid[min(uint(naga_mod(index, 5u)), 4u)][min(uint(naga_mod(index, 4u)), 3u)];
    float _e27 = particles[min(uint(naga_mod(index, 2u)), 1u)].position.x;
    float _e37 = particles[min(uint(naga_mod(index, 2u)), 1u)].history[min(uint(naga_mod(index, 4u)), 3u)];
    float sum = ((((float(_e5) + _e10) + _e19) + _e27) + _e37);
    uint _e43; InterlockedAdd(particles[0].hits, naga_f2u32(sum), _e43);
    if ((index == 0u)) {
        uint _e50; InterlockedOr(particles[1].hits, 0, _e50);
        total = _e50;
        return;
    } else {
        return;
    }
}

[numthreads(64, 1, 1)]
void no_index(uint local_invocation_index : SV_GroupIndex)
{
    if ((local_invocation_index == 0u)) {
        total = 0u;
    }
    GroupMemoryBarrierWithGroupSync();
    uint phony = total;
    return;
}

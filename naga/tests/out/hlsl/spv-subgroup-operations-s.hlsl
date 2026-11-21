static uint global = (uint)0;
static uint global_1 = (uint)0;
static uint global_2 = (uint)0;
static uint global_3 = (uint)0;

struct ComputeInput_main {
    uint local_invocation_index : SV_GroupIndex;
};

[numthreads(1, 1, 1)]
void main(ComputeInput_main computeinput_main)
{
    uint param = (1u + WaveGetLaneCount() - 1u) / WaveGetLaneCount();
    uint param_1 = computeinput_main.local_invocation_index / WaveGetLaneCount();
    uint param_2 = WaveGetLaneCount();
    uint param_3 = WaveGetLaneIndex();
    global = param;
    global_1 = param_1;
    global_2 = param_2;
    global_3 = param_3;
    uint _e13 = global_2;
    uint _e14 = global_3;
    const uint4 _e17 = WaveActiveBallot(((_e14 & 1u) == 1u));
    const uint4 _e18 = WaveActiveBallot(true);
    const bool _e20 = WaveActiveAllTrue((_e14 != 0u));
    const bool _e22 = WaveActiveAnyTrue((_e14 == 0u));
    const uint _e23 = WaveActiveSum(_e14);
    const uint _e24 = WaveActiveProduct(_e14);
    const uint _e25 = WaveActiveMin(_e14);
    const uint _e26 = WaveActiveMax(_e14);
    const uint _e27 = WaveActiveBitAnd(_e14);
    const uint _e28 = WaveActiveBitOr(_e14);
    const uint _e29 = WaveActiveBitXor(_e14);
    const uint _e30 = WavePrefixSum(_e14);
    const uint _e31 = WavePrefixProduct(_e14);
    const uint _e32 = _e14 + WavePrefixSum(_e14);
    const uint _e33 = _e14 * WavePrefixProduct(_e14);
    const uint _e34 = WaveReadLaneFirst(_e14);
    const uint _e35 = WaveReadLaneAt(_e14, 4u);
    const uint _e38 = WaveReadLaneAt(_e14, ((_e13 - 1u) - _e14));
    const uint _e39 = WaveReadLaneAt(_e14, WaveGetLaneIndex() + 1u);
    const uint _e40 = WaveReadLaneAt(_e14, WaveGetLaneIndex() - 1u);
    const uint _e42 = WaveReadLaneAt(_e14, WaveGetLaneIndex() ^ (_e13 - 1u));
}

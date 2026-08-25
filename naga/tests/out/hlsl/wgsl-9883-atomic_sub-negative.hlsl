groupshared int a;

[numthreads(1, 1, 1)]
void main(uint local_invocation_index : SV_GroupIndex)
{
    if (local_invocation_index == 0) {
        a = (int)0;
    }
    GroupMemoryBarrierWithGroupSync();
    int _e2; InterlockedAdd(a, -int(-1), _e2);
    return;
}

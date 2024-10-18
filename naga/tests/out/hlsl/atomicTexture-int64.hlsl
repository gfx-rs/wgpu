struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

RWTexture2D<uint> image : register(u0);

int ZeroValueint() {
    return (int)0;
}

[numthreads(2, 1, 1)]
void cs_main(uint3 id : SV_GroupThreadID)
{
    InterlockedMax(image[int2(0, 0)],1uL);
    GroupMemoryBarrierWithGroupSync();
    InterlockedMin(image[int2(0, 0)],1uL);
    return;
}

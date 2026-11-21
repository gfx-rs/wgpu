struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

RWTexture2DArray<uint> image_u : register(u0);
RWTexture2DArray<int> image_s : register(u1);

[numthreads(2, 1, 1)]
void cs_main(uint3 id : SV_GroupThreadID)
{
    InterlockedMax(image_u[int3(int2(int(0), int(0)), int(1))],1u);
    InterlockedMin(image_s[int3(int2(int(0), int(0)), 1u)],int(1));
    return;
}

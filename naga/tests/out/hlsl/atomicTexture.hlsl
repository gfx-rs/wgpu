struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

RWTexture2D<uint> image_u : register(u0);
RWTexture2D<int> image_s : register(u1);

[numthreads(2, 1, 1)]
void cs_main(uint3 id : SV_GroupThreadID)
{
    InterlockedMax(image_u[int2(0, 0)],1u);
    InterlockedMin(image_u[int2(0, 0)],1u);
    InterlockedAdd(image_u[int2(0, 0)],1u);
    InterlockedAnd(image_u[int2(0, 0)],1u);
    InterlockedOr(image_u[int2(0, 0)],1u);
    InterlockedXor(image_u[int2(0, 0)],1u);
    InterlockedMax(image_s[int2(0, 0)],1);
    InterlockedMin(image_s[int2(0, 0)],1);
    InterlockedAdd(image_s[int2(0, 0)],1);
    InterlockedAnd(image_s[int2(0, 0)],1);
    InterlockedOr(image_s[int2(0, 0)],1);
    InterlockedXor(image_s[int2(0, 0)],1);
    return;
}

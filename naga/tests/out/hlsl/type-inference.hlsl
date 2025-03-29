int4 ZeroValueint4() {
    return (int4)0;
}

static const uint g1_ = 1u;
static const float g3_ = 1.0;
static const int4 g4_ = ZeroValueint4();
static const int4 g5_ = (int(1)).xxxx;
static const float2x2 g6_ = float2x2(float2(0.0, 0.0), float2(0.0, 0.0));

[numthreads(1, 1, 1)]
void main()
{
    int g0x = int(1);
    float g2x = 1.0;
    float2x2 g7x = float2x2(float2(1.0, 1.0), float2(1.0, 1.0));
    int c0x = int(1);
    uint c1x = 1u;
    float c2x = 1.0;
    float c3x = 1.0;
    int4 c4x = ZeroValueint4();
    int4 c5x = (int(1)).xxxx;
    float2x2 c6x = float2x2(float2(0.0, 0.0), float2(0.0, 0.0));
    float2x2 c7x = float2x2(float2(1.0, 1.0), float2(1.0, 1.0));
    int l0x = (int)0;
    uint l1x = (uint)0;
    float l2x = (float)0;
    float l3x = (float)0;
    int4 l4x = (int4)0;
    int v0_ = int(1);
    uint v1_ = 1u;
    float v2_ = 1.0;
    float v3_ = 1.0;
    int4 v4_ = ZeroValueint4();
    int4 v5_ = (int(1)).xxxx;
    float2x2 v6_ = float2x2(float2(0.0, 0.0), float2(0.0, 0.0));
    float2x2 v7_ = float2x2(float2(1.0, 1.0), float2(1.0, 1.0));

    int4 l5_ = (int(1)).xxxx;
    float2x2 l6_ = float2x2(float2(0.0, 0.0), float2(0.0, 0.0));
    float2x2 l7_ = float2x2(float2(1.0, 1.0), float2(1.0, 1.0));
    l0x = int(1);
    l1x = 1u;
    l2x = 1.0;
    l3x = 1.0;
    l4x = ZeroValueint4();
    return;
}

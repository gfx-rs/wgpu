
struct gl_PerVertex {
    float4 gl_Position : SV_Position;
    float gl_PointSize : PSIZE;
    float gl_ClipDistance[1] : SV_ClipDistance;
    float gl_CullDistance[1] : SV_CullDistance;
};

struct type9 {
    linear float2 member : LOC0;
    float4 gl_Position : SV_Position;
};

static float2 v_uv = (float2)0;
static float2 a_uv1 = (float2)0;
static gl_PerVertex perVertexStruct = { float4(0.0, 0.0, 0.0, 1.0), 1.0, { 0.0 }, { 0.0 } };
static float2 a_pos1 = (float2)0;

struct VertexOutput_main {
    float2 member : LOC0;
    float4 gl_Position : SV_Position;
};

void main1()
{
    float2 _expr12 = a_uv1;
    v_uv = _expr12;
    float2 _expr13 = a_pos1;
    perVertexStruct.gl_Position = float4(_expr13.x, _expr13.y, 0.0, 1.0);
    return;
}

type9 Constructtype9(float2 arg0, float4 arg1) {
    type9 ret;
    ret.member = arg0;
    ret.gl_Position = arg1;
    return ret;
}

VertexOutput_main main(float2 a_uv : LOC1, float2 a_pos : LOC0)
{
    a_uv1 = a_uv;
    a_pos1 = a_pos;
    main1();
    float2 _expr7 = v_uv;
    float4 _expr8 = perVertexStruct.gl_Position;
    const type9 type9 = Constructtype9(_expr7, _expr8);
    const VertexOutput_main type9_1 = { type9.member, type9.gl_Position };
    return type9_1;
}

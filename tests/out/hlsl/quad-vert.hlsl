
struct gl_PerVertex {
    float4 gl_Position : SV_Position;
    float gl_PointSize : PSIZE;
    float gl_ClipDistance[1] : SV_ClipDistance;
    float gl_CullDistance[1] : SV_CullDistance;
};

struct type10 {
    linear float2 member : LOC0;
    float4 gl_Position : SV_Position;
    float gl_PointSize : PSIZE;
    float gl_ClipDistance[1] : SV_ClipDistance;
    float gl_CullDistance[1] : SV_CullDistance;
};

static float2 v_uv = (float2)0;
static float2 a_uv1 = (float2)0;
static gl_PerVertex perVertexStruct = { float4(0.0, 0.0, 0.0, 1.0), 1.0, { 0.0 }, { 0.0 } };
static float2 a_pos1 = (float2)0;

struct VertexOutput_main {
    float2 member : LOC0;
    float4 gl_Position : SV_Position;
    float gl_ClipDistance : SV_ClipDistance;
    float gl_CullDistance : SV_CullDistance;
    float gl_PointSize : PSIZE;
};

void main1()
{
    float2 _expr12 = a_uv1;
    v_uv = _expr12;
    float2 _expr13 = a_pos1;
    perVertexStruct.gl_Position = float4(_expr13.x, _expr13.y, 0.0, 1.0);
    return;
}

VertexOutput_main main(float2 a_uv : LOC1, float2 a_pos : LOC0)
{
    a_uv1 = a_uv;
    a_pos1 = a_pos;
    main1();
    float2 _expr10 = v_uv;
    float4 _expr11 = perVertexStruct.gl_Position;
    float _expr12 = perVertexStruct.gl_PointSize;
    float _expr13[1] = perVertexStruct.gl_ClipDistance;
    float _expr14[1] = perVertexStruct.gl_CullDistance;
    const type10 type10_ = { _expr10, _expr11, _expr12, _expr13, _expr14 };
    const VertexOutput_main type10_1 = { type10_.member, type10_.gl_Position, type10_.gl_ClipDistance, type10_.gl_CullDistance, type10_.gl_PointSize };
    return type10_1;
}

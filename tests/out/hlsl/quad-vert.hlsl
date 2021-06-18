struct gl_PerVertex {
    float4 gl_Position : SV_Position;
    float gl_PointSize : PSIZE;
    float gl_ClipDistance[1] : SV_ClipDistance;
    float gl_CullDistance[1] : SV_CullDistance;
};

struct type10 {
    float2 member : LOC0;
    float4 gl_Position : SV_Position;
    float gl_PointSize : PSIZE;
    float gl_ClipDistance[1] : SV_ClipDistance;
    float gl_CullDistance[1] : SV_CullDistance;
};

static float2 v_uv;
static float2 a_uv1;
static gl_PerVertex perVertexStruct = { float4(0.0, 0.0, 0.0, 1.0), 1.0, float(0.0), float(0.0) };
static float2 a_pos1;

struct VertexInput {
    float2 a_uv2 : LOC1;
    float2 a_pos2 : LOC0;
};

void main1()
{
    float2 _expr12 = a_uv1;
    v_uv = _expr12;
    float2 _expr13 = a_pos1;
    perVertexStruct.gl_Position = float4(_expr13.x, _expr13.y, 0.0, 1.0);
    return;
}

type10 vert_main(VertexInput vertexinput)
{
    a_uv1 = vertexinput.a_uv2;
    a_pos1 = vertexinput.a_pos2;
    main1();
    float2 _expr10 = v_uv;
    float4 _expr11 = perVertexStruct.gl_Position;
    float _expr12 = perVertexStruct.gl_PointSize;
    float _expr13[1] = perVertexStruct.gl_ClipDistance;
    float _expr14[1] = perVertexStruct.gl_CullDistance;
    const type10 type10_ = { _expr10, _expr11, _expr12, _expr13, _expr14 };
    return type10_;
}

struct OutVertex
{
    float4 Position : SV_POSITION;
    float4 Color : COLOR;
};
struct InVertex
{
    float4 Color : COLOR;
};

static const float4 positions[3] = { float4(0., 1.0, 0., 1.0), float4(-1.0, -1.0, 0., 1.0), float4(1.0, -1.0, 0., 1.0) };
static const float4 colors[3] = { float4(0., 1., 0., 1.), float4(0., 0., 1., 1.), float4(1., 0., 0., 1.) };

struct EmptyPayload
{
    uint _nullField;
};
groupshared EmptyPayload _emptyPayload;

[numthreads(1, 1, 1)]
void Task()
{
    DispatchMesh(1, 1, 1, _emptyPayload);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void Mesh(out indices uint3 triangles[1], out vertices OutVertex vertices[3], in payload EmptyPayload _emptyPayload)
{
    SetMeshOutputCounts(3, 1);

    vertices[0].Position = positions[0];
    vertices[1].Position = positions[1];
    vertices[2].Position = positions[2];

    vertices[0].Color = colors[0];
    vertices[1].Color = colors[1];
    vertices[2].Color = colors[2];

    triangles[0] = uint3(0, 1, 2);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void MeshNoTask(out indices uint3 triangles[1], out vertices OutVertex vertices[3])
{
    SetMeshOutputCounts(3, 1);

    vertices[0].Position = positions[0];
    vertices[1].Position = positions[1];
    vertices[2].Position = positions[2];

    vertices[0].Color = colors[0];
    vertices[1].Color = colors[1];
    vertices[2].Color = colors[2];

    triangles[0] = uint3(0, 1, 2);
}

float4 Frag(InVertex vertex) : SV_Target
{
    return vertex.Color;
}

struct OutVertex {
    float4 Position : SV_POSITION;
    float4 Color: COLOR;
};
struct OutPrimitive {
    float4 ColorMask : COLOR_MASK : PRIMITIVE;
    bool CullPrimitive: SV_CullPrimitive;
};
struct InVertex {
    float4 Color: COLOR;
};
struct InPrimitive {
    float4 ColorMask : COLOR_MASK : PRIMITIVE;
};
struct PayloadData {
    float4 ColorMask;
    bool Visible;
};


static const float4 positions[3] = {float4(0., 1.0, 0., 1.0), float4(-1.0, -1.0, 0., 1.0), float4(1.0, -1.0, 0., 1.0)};
static const float4 colors[3] = {float4(0., 1., 0., 1.), float4(0., 0., 1., 1.), float4(1., 0., 0., 1.)};

groupshared PayloadData outPayload;

[numthreads(1, 1, 1)]
void Task() {
    outPayload.ColorMask = float4(1.0, 1.0, 0.0, 1.0);
    outPayload.Visible = true;
    DispatchMesh(3, 1, 1, outPayload);
}

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void Mesh(out indices uint3 triangles[1], out vertices OutVertex vertices[3], out primitives OutPrimitive primitives[1], in payload PayloadData payload) {
    SetMeshOutputCounts(3, 1);

    vertices[0].Position = positions[0];
    vertices[1].Position = positions[1];
    vertices[2].Position = positions[2];
    
    vertices[0].Color = colors[0] * payload.ColorMask;
    vertices[1].Color = colors[1] * payload.ColorMask;
    vertices[2].Color = colors[2] * payload.ColorMask;

    triangles[0] = uint3(0, 1, 2);
    primitives[0].ColorMask = float4(1.0, 0.0, 0.0, 1.0);
    primitives[0].CullPrimitive = !payload.Visible;
}

float4 Frag(InVertex vertex, InPrimitive primitive) : SV_Target  {
    return vertex.Color * primitive.ColorMask;
}

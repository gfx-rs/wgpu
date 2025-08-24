using namespace metal;

struct OutVertex {
    float4 Position [[position]];
    float4 Color;
};

struct OutPrimitive {
    float4 ColorMask [[flat]];
    bool CullPrimitive;
};

struct InVertex {
    float4 Color;
};

struct InPrimitive {
    float4 ColorMask [[flat]];
};

struct PayloadData {
    float4 ColorMask;
    bool Visible;
};

using Meshlet = metal::mesh<OutVertex, OutPrimitive, 3, 1, topology::triangle>;


constant float4 positions[3] = {
    float4(0.0, 1.0, 0.0, 1.0),
    float4(-1.0, -1.0, 0.0, 1.0),
    float4(1.0, -1.0, 0.0, 1.0)
};

constant float4 colors[3] = {
    float4(0.0, 1.0, 0.0, 1.0),
    float4(0.0, 0.0, 1.0, 1.0),
    float4(1.0, 0.0, 0.0, 1.0)
};


[[object]]
void taskShader(uint3 tid [[thread_position_in_grid]], object_data PayloadData &outPayload [[payload]], grid_properties grid) {
    outPayload.ColorMask = float4(1.0, 1.0, 0.0, 1.0);
    outPayload.Visible = true;
    grid.set_threadgroups_per_grid(uint3(3, 1, 1));
}

[[mesh, topology(triangle)]]
void meshShader(
    object_data PayloadData const& payload [[payload]],
    Meshlet out,
)
{
    out.set_primitive_count(1);

    for(int i = 0;i < 3;i++) {
        OutVertex vert;
        vert.Position = positions[i];
        vert.Color = colors[i] * payload.ColorMask;
        mesh.set_vertex(i, vert);
        out.set_index(i, i);
    }

    triangles[0] = uint3(0, 1, 2);
    OutPrimitive prim;
    prim.ColorMask = float4(1.0, 0.0, 0.0, 1.0);
    prim.CullPrimitive = !payload.Visible;
    out.set_primitive(0, prim);
}

fragment float4 fragShader(OutVertex inVertex [[stage_in]], OutPrimitive inPrimitive [[stage_in]]) {
    return inVertex.Color * inPrimitive.ColorMask;
}

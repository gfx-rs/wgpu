struct VertexOutput {
    float4 position : SV_Position;
    float clip_distances[1] : SV_ClipDistance;
};

struct VertexOutputSplit {
    float4 color : LOC0;
    float clip_distances[6] : SV_ClipDistance;
    float4 position : SV_Position;
};

struct VertexOutput_main {
    float4 position : SV_Position;
    float clip_distances : SV_ClipDistance0;
};

struct VertexOutput_main_split {
    float4 color : LOC0;
    float4 position_1 : SV_Position;
    float4 clip_distances_1 : SV_ClipDistance0;
    float2 clip_distances_1_ : SV_ClipDistance1;
};

VertexOutput_main main()
{
    VertexOutput out_ = (VertexOutput)0;

    out_.clip_distances[0] = 0.5;
    VertexOutput _e4 = out_;
    const VertexOutput vertexoutput = _e4;
    const VertexOutput_main vertexoutput_1 = { vertexoutput.position, float(vertexoutput.clip_distances[0]) };
    return vertexoutput_1;
}

VertexOutput_main_split main_split()
{
    VertexOutputSplit out_1 = (VertexOutputSplit)0;
    int i = int(0);

    uint2 loop_bound = uint2(4294967295u, 4294967295u);
    bool loop_init = true;
    while(true) {
        if (all(loop_bound == uint2(0u, 0u))) { break; }
        loop_bound -= uint2(loop_bound.y == 0u, 1u);
        if (!loop_init) {
            int _e14 = i;
            i = asint(asuint(_e14) + asuint(int(1)));
        }
        loop_init = false;
        int _e3 = i;
        if ((_e3 < int(6))) {
        } else {
            break;
        }
        {
            int _e7 = i;
            int _e9 = i;
            out_1.clip_distances[min(uint(_e7), 5u)] = (float(_e9) - 2.5);
        }
    }
    VertexOutputSplit _e16 = out_1;
    const VertexOutputSplit vertexoutputsplit = _e16;
    const VertexOutput_main_split vertexoutputsplit_1 = { vertexoutputsplit.color, vertexoutputsplit.position, float4(vertexoutputsplit.clip_distances[0], vertexoutputsplit.clip_distances[1], vertexoutputsplit.clip_distances[2], vertexoutputsplit.clip_distances[3]), float2(vertexoutputsplit.clip_distances[4], vertexoutputsplit.clip_distances[5]) };
    return vertexoutputsplit_1;
}

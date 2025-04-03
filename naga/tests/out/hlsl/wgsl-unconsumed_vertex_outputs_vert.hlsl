struct VertexOut {
    float4 position : SV_Position;
    float value : LOC1;
    float4 unused_value2_ : LOC2;
    float unused_value : LOC0;
    float value2_ : LOC3;
};

struct VertexOutput_vs_main {
    float value : LOC1;
    float value2_ : LOC3;
    float4 position : SV_Position;
};

VertexOut ConstructVertexOut(float4 arg0, float arg1, float4 arg2, float arg3, float arg4) {
    VertexOut ret = (VertexOut)0;
    ret.position = arg0;
    ret.value = arg1;
    ret.unused_value2_ = arg2;
    ret.unused_value = arg3;
    ret.value2_ = arg4;
    return ret;
}

VertexOutput_vs_main vs_main()
{
    const VertexOut vertexout = ConstructVertexOut((1.0).xxxx, 1.0, (2.0).xxxx, 1.0, 0.5);
    const VertexOutput_vs_main vertexout_1 = { vertexout.value, vertexout.value2_, vertexout.position };
    return vertexout_1;
}

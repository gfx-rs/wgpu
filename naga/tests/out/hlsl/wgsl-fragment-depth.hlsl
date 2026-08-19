struct NagaConstants {
    int first_vertex;
    int first_instance;
    uint other;
};
ConstantBuffer<NagaConstants> _NagaConstants: register(b0, space1);

struct StructDepthOutput {
    float depth : SV_DepthGreaterEqual;
};

float greater() : SV_DepthGreaterEqual
{
    return 0.5;
}

float less() : SV_DepthLessEqual
{
    return 0.5;
}

float plain() : SV_Depth
{
    return 0.5;
}

StructDepthOutput ConstructStructDepthOutput(float arg0) {
    StructDepthOutput ret = (StructDepthOutput)0;
    ret.depth = arg0;
    return ret;
}

StructDepthOutput struct_greater()
{
    const StructDepthOutput structdepthoutput = ConstructStructDepthOutput(0.5);
    return structdepthoutput;
}

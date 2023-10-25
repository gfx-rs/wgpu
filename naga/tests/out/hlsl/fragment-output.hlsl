struct FragmentOutputVec4Vec3_ {
    float4 vec4f : SV_Target0;
    nointerpolation int4 vec4i : SV_Target1;
    nointerpolation uint4 vec4u : SV_Target2;
    float3 vec3f : SV_Target3;
    nointerpolation int3 vec3i : SV_Target4;
    nointerpolation uint3 vec3u : SV_Target5;
};

struct FragmentOutputVec2Scalar {
    float2 vec2f : SV_Target0;
    nointerpolation int2 vec2i : SV_Target1;
    nointerpolation uint2 vec2u : SV_Target2;
    float scalarf : SV_Target3;
    nointerpolation int scalari : SV_Target4;
    nointerpolation uint scalaru : SV_Target5;
};

FragmentOutputVec4Vec3_ main_vec4vec3_()
{
    FragmentOutputVec4Vec3_ output = (FragmentOutputVec4Vec3_)0;

    output.vec4f = (0.0).xxxx;
    output.vec4i = (0).xxxx;
    output.vec4u = (0u).xxxx;
    output.vec3f = (0.0).xxx;
    output.vec3i = (0).xxx;
    output.vec3u = (0u).xxx;
    FragmentOutputVec4Vec3_ _expr19 = output;
    const FragmentOutputVec4Vec3_ fragmentoutputvec4vec3_ = _expr19;
    return fragmentoutputvec4vec3_;
}

FragmentOutputVec2Scalar main_vec2scalar()
{
    FragmentOutputVec2Scalar output_1 = (FragmentOutputVec2Scalar)0;

    output_1.vec2f = (0.0).xx;
    output_1.vec2i = (0).xx;
    output_1.vec2u = (0u).xx;
    output_1.scalarf = 0.0;
    output_1.scalari = 0;
    output_1.scalaru = 0u;
    FragmentOutputVec2Scalar _expr16 = output_1;
    const FragmentOutputVec2Scalar fragmentoutputvec2scalar = _expr16;
    return fragmentoutputvec2scalar;
}

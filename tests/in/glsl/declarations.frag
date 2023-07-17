#version 450

layout(location = 0) in VertexData {
    vec2 position;
    vec2 a;
} vert;

layout(location = 0) out FragmentData {
    vec2 position;
    vec2 a;
} frag;

layout(location = 2) in  vec4  in_array[2];
layout(location = 2) out vec4 out_array[2];

struct TestStruct {
    float a;
    float b;
};

float array_2d[2][2];
float array_toomanyd[2][2][2][2][2][2][2];

struct LightScatteringParams {
    float BetaRay, BetaMie[3], HGg, DistanceMul[4], BlendCoeff;
    vec3 SunDirection, SunColor;
};

void main() {
    const vec3 positions[2] = vec3[2](
        vec3(-1.0, 1.0, 0.0),
        vec3(-1.0, -1.0, 0.0)
    );
    const TestStruct strct = TestStruct( 1, 2 );
    const vec4 from_input_array = in_array[1];
    const float a = array_2d[0][0];
    const float b = array_toomanyd[0][0][0][0][0][0][0];
    out_array[0] = vec4(2.0);
}

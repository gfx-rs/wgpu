#version 450

layout(location = 0) in VertexData {
    vec2 position;
    vec2 a;
} vert;

layout(location = 0) out FragmentData {
    vec2 position;
    vec2 a;
} frag;

struct TestStruct {
    float a;
    float b;
};


void main() {
    const vec3 positions[2] = vec3[2](
        vec3(-1.0, 1.0, 0.0),
        vec3(-1.0, -1.0, 0.0)
    );
    const TestStruct strct = TestStruct( 1, 2 );
}

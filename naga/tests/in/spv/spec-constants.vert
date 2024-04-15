#version 450

layout (constant_id = 0) const float TEST_CONSTANT = 64.0;
layout (constant_id = 1) const bool TEST_CONSTANT_TRUE = true;
layout (constant_id = 2) const bool TEST_CONSTANT_FALSE = false;
// layout (constant_id = 3) const vec2 TEST_CONSTANT_COMPOSITE = vec2(TEST_CONSTANT, 3.0);
// glslc error: 'constant_id' : can only be applied to a scalar

layout(location = 0) in vec3 Vertex_Position;
layout(location = 1) in vec3 Vertex_Normal;
layout(location = 2) in vec2 Vertex_Uv;

layout(location = 0) out vec2 v_Uv;

layout(set = 0, binding = 0) uniform Camera {
    mat4 ViewProj;
};
layout(set = 2, binding = 0) uniform Transform {
    mat4 Model;
};
layout(set = 2, binding = 1) uniform Sprite_size {
    vec2 size;
};

void main() {
    float test_constant = TEST_CONSTANT * float(TEST_CONSTANT_TRUE) * float(TEST_CONSTANT_FALSE)
        ;//* TEST_CONSTANT_COMPOSITE.x * TEST_CONSTANT_COMPOSITE.y;
    v_Uv = Vertex_Uv;
    vec3 position = Vertex_Position * vec3(size, 1.0);
    gl_Position = ViewProj * Model * vec4(position, 1.0) * test_constant;
}

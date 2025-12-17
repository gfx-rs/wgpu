#version 450

// Specialization constants with constant_id layout qualifier
layout(constant_id = 0) const bool SPEC_CONST_BOOL = true;
layout(constant_id = 1) const int SPEC_CONST_INT = 42;
layout(constant_id = 2) const uint SPEC_CONST_UINT = 10u;
layout(constant_id = 3) const float SPEC_CONST_FLOAT = 3.14;

// NOTE: Naga does not yet support GLSL const variables depending on specialization constants (constant_id).
// const vec3 scVec = vec3(SPEC_CONST_FLOAT, 1, 1); // Would cause error

layout(location = 0) out vec4 o_color;

void main() {
    float result = 0.0;
    
    if (SPEC_CONST_BOOL) {
        result += float(SPEC_CONST_INT);
    }
    
    result += float(SPEC_CONST_UINT) * SPEC_CONST_FLOAT;
    
    o_color = vec4(result, 0.0, 0.0, 1.0);
}

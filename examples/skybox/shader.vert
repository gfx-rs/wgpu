#version 450

layout(location = 0) out vec3 v_Uv;

layout(set = 0, binding = 0) uniform Data {
  mat4 proj;
  mat4 view;
};

void main() {
    vec4 pos = vec4(0.0);
    switch(gl_VertexIndex) {
        case 0: pos = vec4(-1.0, -1.0, 0.0, 1.0); break;
        case 1: pos = vec4( 3.0, -1.0, 0.0, 1.0); break;
        case 2: pos = vec4(-1.0,  3.0, 0.0, 1.0); break;
    }
    mat3 invModelView = transpose(mat3(view));
    vec3 unProjected = (inverse(proj) * pos).xyz;
    v_Uv = invModelView * unProjected;

    gl_Position = pos;
}

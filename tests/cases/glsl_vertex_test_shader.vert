#version 450 core

layout(location=0) in vec3 a_position;
layout(location=1) in vec3 a_color;
layout(location=2) in vec3 a_normal;
        
layout(location=0) out vec3 v_position;
layout(location=1) out vec3 v_color;
layout(location=2) out vec3 v_normal;
        
layout(set=0, binding=0)
uniform Globals {
    mat4 u_view_proj;
    vec3 u_view_position;
};
        
layout(set=2, binding=0)
uniform Locals {
    mat4 u_transform;
    vec2 U_min_max;
};
        
void main() {
    v_color = a_color;
    v_normal = a_normal;
        
    v_position = (u_transform * vec4(a_position, 1.0)).xyz;
    gl_Position = u_view_proj * u_transform * vec4(a_position, 1.0);
}

#version 450 core

layout(location=0) in vec3 v_position;
layout(location=1) in vec3 v_color;
layout(location=2) in vec3 v_normal;
        
layout(location=0) out vec4 f_color;
        
layout(set=0, binding=0)
uniform Globals {
    mat4 u_view_proj;
    vec3 u_view_position;
};
        
layout(set = 1, binding = 0) uniform Light {
    vec3 u_position;
    vec3 u_color;
};
        
layout(set=2, binding=0)
uniform Locals {
    mat4 u_transform;
    vec2 u_min_max;
};
        
layout (set = 2, binding = 1) uniform texture2D t_color;
layout (set = 2, binding = 2) uniform sampler s_color;
        
float invLerp(float from, float to, float value){
    return (value - from) / (to - from);
}
        
void main() {
    vec3 object_color = 
        texture(sampler2D(t_color,s_color), vec2(invLerp(u_min_max.x,u_min_max.y,length(v_position)),0.0)).xyz;
        
    float ambient_strength = 0.1;
    vec3 ambient_color = u_color * ambient_strength;
        
    vec3 normal = normalize(v_normal);
    vec3 light_dir = normalize(u_position - v_position);
        
    float diffuse_strength = max(dot(normal, light_dir), 0.0);
    vec3 diffuse_color = u_color * diffuse_strength;
        
    vec3 view_dir = normalize(u_view_position - v_position);
    vec3 half_dir = normalize(view_dir + light_dir);
        
    float specular_strength = pow(max(dot(normal, half_dir), 0.0), 32);
        
    vec3 specular_color = specular_strength * u_color;
        
    vec3 result = (ambient_color + diffuse_color + specular_color) * object_color;
        
    f_color = vec4(result, 1.0);
}

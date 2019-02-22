#version 450

const int MAX_LIGHTS = 10;

layout(location = 0) in vec3 v_Normal;
layout(location = 1) in vec4 v_Position;

layout(location = 0) out vec4 o_Target;

struct Light {
    mat4 proj;
    vec4 pos;
    vec4 color;
};

layout(set = 0, binding = 0) uniform Globals {
    mat4 u_ViewProj;
    uvec4 u_NumLights;
};
layout(set = 0, binding = 1) uniform Lights {
    Light u_Lights[MAX_LIGHTS];
};
layout(set = 0, binding = 2) uniform texture2DArray t_Shadow;
layout(set = 0, binding = 3) uniform samplerShadow s_Shadow;

layout(set = 1, binding = 0) uniform Entity {
    mat4 u_World;
    vec4 u_Color;
};


void main() {
    vec3 normal = normalize(v_Normal);
    vec3 ambient = vec3(0.05, 0.05, 0.05);
    // accumulate color
    vec3 color = ambient;
    for (int i=0; i<int(u_NumLights.x) && i<MAX_LIGHTS; ++i) {
        Light light = u_Lights[i];
        // project into the light space
        vec4 light_local = light.proj * v_Position;
        // compute texture coordinates for shadow lookup
        light_local.y *= -1.0; // difference in Vulkan target versus texture coordinates...
        light_local.xyw = (light_local.xyz/light_local.w + 1.0) / 2.0;
        light_local.z = i;
        // do the lookup, using HW PCF and comparison
        float shadow = texture(sampler2DArrayShadow(t_Shadow, s_Shadow), light_local);
        // compute Lambertian diffuse term
        vec3 light_dir = normalize(light.pos.xyz - v_Position.xyz);
        float diffuse = max(0.0, dot(normal, light_dir));
        // add light contribution
        color += shadow * diffuse * light.color.xyz;
    }
    // multiply the light by material color
    o_Target = vec4(color, 1.0) * u_Color;
}

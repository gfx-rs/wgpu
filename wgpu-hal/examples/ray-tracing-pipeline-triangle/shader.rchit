// glslc --target-spv=spv1.6 shader.rchit -o shader.rchit.spv
#version 460 core
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_scalar_block_layout : require

layout(set = 0, binding = 2) uniform accelerationStructureEXT tlas;

hitAttributeEXT vec2 barycentric_coord; 

layout(shaderRecordEXT, scalar) buffer shader_record {
	vec4 col;
}record;

struct ray_payload_struct {
    vec3 pos;
    vec3 dir;
    vec3 col;
};

layout (location = 0) rayPayloadInEXT ray_payload_struct ray_payload;

struct call_payload_struct {
    vec3 col;
};

layout(location = 1) callableDataEXT call_payload_struct call_payload;


vec2 bary_lerp2(vec2 a, vec2 b, vec2 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec3 bary_lerp3(vec3 a, vec3 b, vec3 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

vec4 bary_lerp4(vec4 a, vec4 b, vec4 c, vec3 barycentrics) {
    return a * barycentrics.x + b * barycentrics.y + c * barycentrics.z;
}

void main() {
    vec3 barycentrics = vec3(1.0f - barycentric_coord.x - barycentric_coord.y, barycentric_coord.x, barycentric_coord.y);
    vec3 col = bary_lerp3(vec3(1,0,0),vec3(0,1,0),vec3(0,0,1), barycentrics); 

    call_payload.col = col;

    if(gl_InstanceCustomIndexEXT == 1){
        executeCallableEXT(
            0, // SBT callable index
            1 // payload location
        );
    }

    ray_payload.col = mix(call_payload.col,record.col.rgb,record.col.w);
}

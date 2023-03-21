// glslc --target-spv=spv1.6 shader.rmiss -o shader.rmiss.spv
#version 460 core
#extension GL_EXT_ray_tracing : require

const float PI = 3.14159265;
const float INV_PI = 1.0 / PI;
const float INV_2PI = 0.5 / PI;

struct ray_payload {
    vec3 pos;
    vec3 dir;
    vec3 col;
};
layout (location = 0) rayPayloadInEXT ray_payload payload;

vec2 dir_to_uv(vec3 direction)
{
    vec2 uv = vec2(atan(direction.z, direction.x), asin(-direction.y));
    uv = vec2(uv.x * INV_2PI, uv.y * INV_PI) + 0.5;
    return uv;
}

void main() {
    payload.col = vec3(dir_to_uv(normalize(payload.dir)),1.);
}

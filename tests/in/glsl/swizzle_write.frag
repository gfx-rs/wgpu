#version 450

void foo(inout vec2 p) {}

void main() {
    vec3 x = vec3(2.0);
    x.zxy.xy = vec2(3.0, 4.0);
    x.rg *= 5.0;
    x.zy++;
    foo(x.xz);
}

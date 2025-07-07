// language: metal1.2
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;


void color(
    metal::texture2d<float, metal::access::sample> t,
    metal::sampler s
) {
    metal::float4 phony = t.sample(s, metal::float2(1.0, 2.0));
    metal::float4 phony_1 = t.sample(s, metal::float2(1.0, 2.0), metal::int2(3, 4));
    metal::float4 phony_2 = t.sample(s, metal::float2(1.0, 2.0), metal::level(0.0));
    metal::float4 phony_3 = t.sample(s, metal::float2(1.0, 2.0), metal::level(0.0));
    metal::float4 phony_4 = t.sample(s, metal::float2(1.0, 2.0), metal::gradient2d(metal::float2(3.0, 4.0), metal::float2(5.0, 6.0)));
    metal::float4 phony_5 = t.sample(s, metal::float2(1.0, 2.0), metal::bias(1.0));
    return;
}

void depth(
    metal::sampler s,
    metal::depth2d<float, metal::access::sample> d,
    metal::sampler c
) {
    float phony_6 = d.sample(s, metal::float2(1.0, 2.0), metal::level(1));
    float phony_7 = d.sample_compare(c, metal::float2(1.0, 2.0), 0.0);
    metal::float4 phony_8 = d.gather_compare(c, metal::float2(1.0, 2.0), 0.0);
    return;
}

void storage(
    metal::texture2d<float, metal::access::read_write> st
) {
    st.write(metal::float4(2.0, 3.0, 4.0, 5.0), metal::uint2(metal::int2(0, 1)));
    return;
}

fragment void main_(
  metal::texture2d<float, metal::access::sample> t [[user(fake0)]]
, metal::sampler s [[user(fake0)]]
, metal::depth2d<float, metal::access::sample> d [[user(fake0)]]
, metal::sampler c [[user(fake0)]]
, metal::texture2d<float, metal::access::read_write> st [[user(fake0)]]
) {
    color(t, s);
    depth(s, d, c);
    storage(st);
    return;
}

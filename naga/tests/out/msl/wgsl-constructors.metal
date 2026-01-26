// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct Foo {
    metal::float4 a;
    int b;
    char _pad2[12];
};
struct type_6 {
    metal::float2x2 inner[1];
};
struct type_10 {
    Foo inner[3];
};
struct type_12 {
    int inner[4];
};
constant metal::float3 const1_ = metal::float3(0.0);
constant metal::float2x2 const3_ = metal::float2x2(metal::float2(0.0, 1.0), metal::float2(2.0, 3.0));
constant type_6 const4_ = type_6 {metal::float2x2(metal::float2(0.0, 1.0), metal::float2(2.0, 3.0))};
constant bool cz0_ = bool {};
constant int cz1_ = int {};
constant uint cz2_ = uint {};
constant float cz3_ = float {};
constant metal::uint2 cz4_ = metal::uint2 {};
constant metal::float2x2 cz5_ = metal::float2x2 {};
constant type_10 cz6_ = type_10 {};
constant Foo cz7_ = Foo {};
constant metal::uint2 cp1_ = metal::uint2(0u);

kernel void main_(
) {
    Foo foo = {};
    foo = Foo {metal::float4(1.0), 1};
    metal::float2x2 m0_ = metal::float2x2(metal::float2(1.0, 0.0), metal::float2(0.0, 1.0));
    metal::float4x4 m1_ = metal::float4x4(metal::float4(1.0, 0.0, 0.0, 0.0), metal::float4(0.0, 1.0, 0.0, 0.0), metal::float4(0.0, 0.0, 1.0, 0.0), metal::float4(0.0, 0.0, 0.0, 1.0));
    metal::uint2 zvc8_ = metal::uint2(0u, 0u);
    metal::float2 zvc9_ = metal::float2(0.0, 0.0);
    metal::uint2 cit0_ = metal::uint2(0u);
    metal::float2x2 cit1_ = metal::float2x2(metal::float2(0.0), metal::float2(0.0));
    type_12 cit2_ = type_12 {0, 1, 2, 3};
    metal::uint2 ic4_ = metal::uint2(0u, 0u);
    metal::float2x3 ic5_ = metal::float2x3(metal::float3(0.0, 0.0, 0.0), metal::float3(0.0, 0.0, 0.0));
    return;
}

// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct NoPadding {
    metal::packed_float3 v3_;
    float f3_;
};
struct NeedsPadding {
    float f3_forces_padding;
    char _pad1[12];
    metal::packed_float3 v3_needs_padding;
    float f3_;
};

struct no_padding_fragInput {
    metal::float3 v3_ [[user(loc0), center_perspective]];
    float f3_ [[user(loc1), center_perspective]];
};
struct no_padding_fragOutput {
    metal::float4 member [[color(0)]];
};
fragment no_padding_fragOutput no_padding_frag(
  no_padding_fragInput varyings [[stage_in]]
) {
    const NoPadding input = { varyings.v3_, varyings.f3_ };
    return no_padding_fragOutput { metal::float4(0.0) };
}


struct no_padding_vertInput {
    metal::float3 v3_ [[attribute(0)]];
    float f3_ [[attribute(1)]];
};
struct no_padding_vertOutput {
    metal::float4 member_1 [[position]];
};
vertex no_padding_vertOutput no_padding_vert(
  no_padding_vertInput varyings_1 [[stage_in]]
) {
    const NoPadding input_1 = { varyings_1.v3_, varyings_1.f3_ };
    return no_padding_vertOutput { metal::float4(0.0) };
}


kernel void no_padding_comp(
  constant NoPadding& no_padding_uniform [[user(fake0)]]
, device NoPadding const& no_padding_storage [[user(fake0)]]
) {
    NoPadding x = {};
    NoPadding _e2 = no_padding_uniform;
    x = _e2;
    NoPadding _e4 = no_padding_storage;
    x = _e4;
    return;
}


struct needs_padding_fragInput {
    float f3_forces_padding [[user(loc0), center_perspective]];
    metal::float3 v3_needs_padding [[user(loc1), center_perspective]];
    float f3_ [[user(loc2), center_perspective]];
};
struct needs_padding_fragOutput {
    metal::float4 member_3 [[color(0)]];
};
fragment needs_padding_fragOutput needs_padding_frag(
  needs_padding_fragInput varyings_3 [[stage_in]]
) {
    const NeedsPadding input_2 = { varyings_3.f3_forces_padding, {}, varyings_3.v3_needs_padding, varyings_3.f3_ };
    return needs_padding_fragOutput { metal::float4(0.0) };
}


struct needs_padding_vertInput {
    float f3_forces_padding [[attribute(0)]];
    metal::float3 v3_needs_padding [[attribute(1)]];
    float f3_ [[attribute(2)]];
};
struct needs_padding_vertOutput {
    metal::float4 member_4 [[position]];
};
vertex needs_padding_vertOutput needs_padding_vert(
  needs_padding_vertInput varyings_4 [[stage_in]]
) {
    const NeedsPadding input_3 = { varyings_4.f3_forces_padding, {}, varyings_4.v3_needs_padding, varyings_4.f3_ };
    return needs_padding_vertOutput { metal::float4(0.0) };
}


kernel void needs_padding_comp(
  constant NeedsPadding& needs_padding_uniform [[user(fake0)]]
, device NeedsPadding const& needs_padding_storage [[user(fake0)]]
) {
    NeedsPadding x_1 = {};
    NeedsPadding _e2 = needs_padding_uniform;
    x_1 = _e2;
    NeedsPadding _e4 = needs_padding_storage;
    x_1 = _e4;
    return;
}

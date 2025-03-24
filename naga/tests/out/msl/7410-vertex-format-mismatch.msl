// language: metal1.0
#include <metal_stdlib>
#include <simd/simd.h>

using metal::uint;

struct _mslBufferSizes {
    uint buffer_size1;
};

struct Out {
    metal::float4 position;
    metal::float4 color;
    metal::float2 texcoord;
};
metal::float3 unpackFloat32x3_(uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6, uint b7, uint b8, uint b9, uint b10, uint b11) {
    return metal::float3(as_type<float>(b3 << 24 | b2 << 16 | b1 << 8 | b0), as_type<float>(b7 << 24 | b6 << 16 | b5 << 8 | b4), as_type<float>(b11 << 24 | b10 << 16 | b9 << 8 | b8));
}
metal::float4 unpackFloat32x4_(uint b0, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6, uint b7, uint b8, uint b9, uint b10, uint b11, uint b12, uint b13, uint b14, uint b15) {
    return metal::float4(as_type<float>(b3 << 24 | b2 << 16 | b1 << 8 | b0), as_type<float>(b7 << 24 | b6 << 16 | b5 << 8 | b4), as_type<float>(b11 << 24 | b10 << 16 | b9 << 8 | b8), as_type<float>(b15 << 24 | b14 << 16 | b13 << 8 | b12));
}

struct main_Output {
    metal::float4 position [[position]];
    metal::float4 color [[user(loc0), center_perspective]];
    metal::float2 texcoord [[user(loc1), center_perspective]];
};
struct vb_1_type { metal::uchar data[48]; };
vertex main_Output main_(
  uint v_id [[vertex_id]]
, const device vb_1_type* vb_1_in [[buffer(1)]]
, constant _mslBufferSizes& _buffer_sizes [[user(fake0)]]
) {
    metal::float4 position = {};
    metal::float4 color = {};
    metal::float2 texcoord = {};
    if (v_id < (_buffer_sizes.buffer_size1 / 48)) {
        const vb_1_type vb_1_elem = vb_1_in[v_id];
        position = metal::float4(unpackFloat32x3_(vb_1_elem.data[0], vb_1_elem.data[1], vb_1_elem.data[2], vb_1_elem.data[3], vb_1_elem.data[4], vb_1_elem.data[5], vb_1_elem.data[6], vb_1_elem.data[7], vb_1_elem.data[8], vb_1_elem.data[9], vb_1_elem.data[10], vb_1_elem.data[11]), 1.0);
        color = unpackFloat32x4_(vb_1_elem.data[16], vb_1_elem.data[17], vb_1_elem.data[18], vb_1_elem.data[19], vb_1_elem.data[20], vb_1_elem.data[21], vb_1_elem.data[22], vb_1_elem.data[23], vb_1_elem.data[24], vb_1_elem.data[25], vb_1_elem.data[26], vb_1_elem.data[27], vb_1_elem.data[28], vb_1_elem.data[29], vb_1_elem.data[30], vb_1_elem.data[31]);
        texcoord = metal::float2(unpackFloat32x3_(vb_1_elem.data[32], vb_1_elem.data[33], vb_1_elem.data[34], vb_1_elem.data[35], vb_1_elem.data[36], vb_1_elem.data[37], vb_1_elem.data[38], vb_1_elem.data[39], vb_1_elem.data[40], vb_1_elem.data[41], vb_1_elem.data[42], vb_1_elem.data[43]));
    }
    const auto _tmp = Out {position, color, texcoord};
    return main_Output { _tmp.position, _tmp.color, _tmp.texcoord };
}

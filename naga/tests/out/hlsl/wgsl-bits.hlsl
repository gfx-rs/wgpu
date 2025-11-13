int naga_insertBits(
    int e,
    int newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
int2 naga_insertBits(
    int2 e,
    int2 newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
int3 naga_insertBits(
    int3 e,
    int3 newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
int4 naga_insertBits(
    int4 e,
    int4 newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
uint naga_insertBits(
    uint e,
    uint newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
uint2 naga_insertBits(
    uint2 e,
    uint2 newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
uint3 naga_insertBits(
    uint3 e,
    uint3 newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
uint4 naga_insertBits(
    uint4 e,
    uint4 newbits,
    uint offset,
    uint count
) {
    uint w = 32u;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    uint mask = ((4294967295u >> (32u - c)) << o);
    return (c == 0 ? e : ((e & ~mask) | ((newbits << o) & mask)));
}
int naga_extractBits(
    int e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
int2 naga_extractBits(
    int2 e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
int3 naga_extractBits(
    int3 e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
int4 naga_extractBits(
    int4 e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
uint naga_extractBits(
    uint e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
uint2 naga_extractBits(
    uint2 e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
uint3 naga_extractBits(
    uint3 e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
uint4 naga_extractBits(
    uint4 e,
    uint offset,
    uint count
) {
    uint w = 32;
    uint o = min(offset, w);
    uint c = min(count, w - o);
    return (c == 0 ? 0 : (e << (w - c - o)) >> (w - c));
}
[numthreads(1, 1, 1)]
void main()
{
    int i = int(0);
    int2 i2_ = (int(0)).xx;
    int3 i3_ = (int(0)).xxx;
    int4 i4_ = (int(0)).xxxx;
    uint u = 0u;
    uint2 u2_ = (0u).xx;
    uint3 u3_ = (0u).xxx;
    uint4 u4_ = (0u).xxxx;
    float2 f2_ = (0.0).xx;
    float4 f4_ = (0.0).xxxx;

    float4 _e28 = f4_;
    u = uint((int(round(clamp(_e28[0], -1.0, 1.0) * 127.0)) & 0xFF) | ((int(round(clamp(_e28[1], -1.0, 1.0) * 127.0)) & 0xFF) << 8) | ((int(round(clamp(_e28[2], -1.0, 1.0) * 127.0)) & 0xFF) << 16) | ((int(round(clamp(_e28[3], -1.0, 1.0) * 127.0)) & 0xFF) << 24));
    float4 _e30 = f4_;
    u = (uint(round(clamp(_e30[0], 0.0, 1.0) * 255.0)) | uint(round(clamp(_e30[1], 0.0, 1.0) * 255.0)) << 8 | uint(round(clamp(_e30[2], 0.0, 1.0) * 255.0)) << 16 | uint(round(clamp(_e30[3], 0.0, 1.0) * 255.0)) << 24);
    float2 _e32 = f2_;
    u = uint((int(round(clamp(_e32[0], -1.0, 1.0) * 32767.0)) & 0xFFFF) | ((int(round(clamp(_e32[1], -1.0, 1.0) * 32767.0)) & 0xFFFF) << 16));
    float2 _e34 = f2_;
    u = (uint(round(clamp(_e34[0], 0.0, 1.0) * 65535.0)) | uint(round(clamp(_e34[1], 0.0, 1.0) * 65535.0)) << 16);
    float2 _e36 = f2_;
    u = (f32tof16(_e36[0]) | f32tof16(_e36[1]) << 16);
    int4 _e38 = i4_;
    u = uint((_e38[0] & 0xFF) | ((_e38[1] & 0xFF) << 8) | ((_e38[2] & 0xFF) << 16) | ((_e38[3] & 0xFF) << 24));
    uint4 _e40 = u4_;
    u = (_e40[0] & 0xFF) | ((_e40[1] & 0xFF) << 8) | ((_e40[2] & 0xFF) << 16) | ((_e40[3] & 0xFF) << 24);
    int4 _e42 = i4_;
    u = uint((clamp(_e42, -128, 127)[0] & 0xFF) | ((clamp(_e42, -128, 127)[1] & 0xFF) << 8) | ((clamp(_e42, -128, 127)[2] & 0xFF) << 16) | ((clamp(_e42, -128, 127)[3] & 0xFF) << 24));
    uint4 _e44 = u4_;
    u = (clamp(_e44, 0, 255)[0] & 0xFF) | ((clamp(_e44, 0, 255)[1] & 0xFF) << 8) | ((clamp(_e44, 0, 255)[2] & 0xFF) << 16) | ((clamp(_e44, 0, 255)[3] & 0xFF) << 24);
    uint _e46 = u;
    f4_ = (float4(int4(_e46 << 24, _e46 << 16, _e46 << 8, _e46) >> 24) / 127.0);
    uint _e48 = u;
    f4_ = (float4(_e48 & 0xFF, _e48 >> 8 & 0xFF, _e48 >> 16 & 0xFF, _e48 >> 24) / 255.0);
    uint _e50 = u;
    f2_ = (float2(int2(_e50 << 16, _e50) >> 16) / 32767.0);
    uint _e52 = u;
    f2_ = (float2(_e52 & 0xFFFF, _e52 >> 16) / 65535.0);
    uint _e54 = u;
    f2_ = float2(f16tof32(_e54), f16tof32((_e54) >> 16));
    uint _e56 = u;
    i4_ = (int4(_e56, _e56 >> 8, _e56 >> 16, _e56 >> 24) << 24 >> 24);
    uint _e58 = u;
    u4_ = (uint4(_e58, _e58 >> 8, _e58 >> 16, _e58 >> 24) << 24 >> 24);
    int _e60 = i;
    int _e61 = i;
    i = naga_insertBits(_e60, _e61, 5u, 10u);
    int2 _e65 = i2_;
    int2 _e66 = i2_;
    i2_ = naga_insertBits(_e65, _e66, 5u, 10u);
    int3 _e70 = i3_;
    int3 _e71 = i3_;
    i3_ = naga_insertBits(_e70, _e71, 5u, 10u);
    int4 _e75 = i4_;
    int4 _e76 = i4_;
    i4_ = naga_insertBits(_e75, _e76, 5u, 10u);
    uint _e80 = u;
    uint _e81 = u;
    u = naga_insertBits(_e80, _e81, 5u, 10u);
    uint2 _e85 = u2_;
    uint2 _e86 = u2_;
    u2_ = naga_insertBits(_e85, _e86, 5u, 10u);
    uint3 _e90 = u3_;
    uint3 _e91 = u3_;
    u3_ = naga_insertBits(_e90, _e91, 5u, 10u);
    uint4 _e95 = u4_;
    uint4 _e96 = u4_;
    u4_ = naga_insertBits(_e95, _e96, 5u, 10u);
    int _e100 = i;
    i = naga_extractBits(_e100, 5u, 10u);
    int2 _e104 = i2_;
    i2_ = naga_extractBits(_e104, 5u, 10u);
    int3 _e108 = i3_;
    i3_ = naga_extractBits(_e108, 5u, 10u);
    int4 _e112 = i4_;
    i4_ = naga_extractBits(_e112, 5u, 10u);
    uint _e116 = u;
    u = naga_extractBits(_e116, 5u, 10u);
    uint2 _e120 = u2_;
    u2_ = naga_extractBits(_e120, 5u, 10u);
    uint3 _e124 = u3_;
    u3_ = naga_extractBits(_e124, 5u, 10u);
    uint4 _e128 = u4_;
    u4_ = naga_extractBits(_e128, 5u, 10u);
    int _e132 = i;
    i = asint(firstbitlow(_e132));
    uint2 _e134 = u2_;
    u2_ = firstbitlow(_e134);
    int3 _e136 = i3_;
    i3_ = asint(firstbithigh(_e136));
    uint3 _e138 = u3_;
    u3_ = firstbithigh(_e138);
    int _e140 = i;
    i = asint(firstbithigh(_e140));
    uint _e142 = u;
    u = firstbithigh(_e142);
    int _e144 = i;
    i = asint(countbits(asuint(_e144)));
    int2 _e146 = i2_;
    i2_ = asint(countbits(asuint(_e146)));
    int3 _e148 = i3_;
    i3_ = asint(countbits(asuint(_e148)));
    int4 _e150 = i4_;
    i4_ = asint(countbits(asuint(_e150)));
    uint _e152 = u;
    u = countbits(_e152);
    uint2 _e154 = u2_;
    u2_ = countbits(_e154);
    uint3 _e156 = u3_;
    u3_ = countbits(_e156);
    uint4 _e158 = u4_;
    u4_ = countbits(_e158);
    int _e160 = i;
    i = asint(reversebits(asuint(_e160)));
    int2 _e162 = i2_;
    i2_ = asint(reversebits(asuint(_e162)));
    int3 _e164 = i3_;
    i3_ = asint(reversebits(asuint(_e164)));
    int4 _e166 = i4_;
    i4_ = asint(reversebits(asuint(_e166)));
    uint _e168 = u;
    u = reversebits(_e168);
    uint2 _e170 = u2_;
    u2_ = reversebits(_e170);
    uint3 _e172 = u3_;
    u3_ = reversebits(_e172);
    uint4 _e174 = u4_;
    u4_ = reversebits(_e174);
    return;
}

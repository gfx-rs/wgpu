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
    int i = 0;
    int2 i2_ = (0).xx;
    int3 i3_ = (0).xxx;
    int4 i4_ = (0).xxxx;
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
    uint _e42 = u;
    f4_ = (float4(int4(_e42 << 24, _e42 << 16, _e42 << 8, _e42) >> 24) / 127.0);
    uint _e44 = u;
    f4_ = (float4(_e44 & 0xFF, _e44 >> 8 & 0xFF, _e44 >> 16 & 0xFF, _e44 >> 24) / 255.0);
    uint _e46 = u;
    f2_ = (float2(int2(_e46 << 16, _e46) >> 16) / 32767.0);
    uint _e48 = u;
    f2_ = (float2(_e48 & 0xFFFF, _e48 >> 16) / 65535.0);
    uint _e50 = u;
    f2_ = float2(f16tof32(_e50), f16tof32((_e50) >> 16));
    uint _e52 = u;
    i4_ = (int4(_e52, _e52 >> 8, _e52 >> 16, _e52 >> 24) << 24 >> 24);
    uint _e54 = u;
    u4_ = (uint4(_e54, _e54 >> 8, _e54 >> 16, _e54 >> 24) << 24 >> 24);
    int _e56 = i;
    int _e57 = i;
    i = naga_insertBits(_e56, _e57, 5u, 10u);
    int2 _e61 = i2_;
    int2 _e62 = i2_;
    i2_ = naga_insertBits(_e61, _e62, 5u, 10u);
    int3 _e66 = i3_;
    int3 _e67 = i3_;
    i3_ = naga_insertBits(_e66, _e67, 5u, 10u);
    int4 _e71 = i4_;
    int4 _e72 = i4_;
    i4_ = naga_insertBits(_e71, _e72, 5u, 10u);
    uint _e76 = u;
    uint _e77 = u;
    u = naga_insertBits(_e76, _e77, 5u, 10u);
    uint2 _e81 = u2_;
    uint2 _e82 = u2_;
    u2_ = naga_insertBits(_e81, _e82, 5u, 10u);
    uint3 _e86 = u3_;
    uint3 _e87 = u3_;
    u3_ = naga_insertBits(_e86, _e87, 5u, 10u);
    uint4 _e91 = u4_;
    uint4 _e92 = u4_;
    u4_ = naga_insertBits(_e91, _e92, 5u, 10u);
    int _e96 = i;
    i = naga_extractBits(_e96, 5u, 10u);
    int2 _e100 = i2_;
    i2_ = naga_extractBits(_e100, 5u, 10u);
    int3 _e104 = i3_;
    i3_ = naga_extractBits(_e104, 5u, 10u);
    int4 _e108 = i4_;
    i4_ = naga_extractBits(_e108, 5u, 10u);
    uint _e112 = u;
    u = naga_extractBits(_e112, 5u, 10u);
    uint2 _e116 = u2_;
    u2_ = naga_extractBits(_e116, 5u, 10u);
    uint3 _e120 = u3_;
    u3_ = naga_extractBits(_e120, 5u, 10u);
    uint4 _e124 = u4_;
    u4_ = naga_extractBits(_e124, 5u, 10u);
    int _e128 = i;
    i = asint(firstbitlow(_e128));
    uint2 _e130 = u2_;
    u2_ = firstbitlow(_e130);
    int3 _e132 = i3_;
    i3_ = asint(firstbithigh(_e132));
    uint3 _e134 = u3_;
    u3_ = firstbithigh(_e134);
    int _e136 = i;
    i = asint(firstbithigh(_e136));
    uint _e138 = u;
    u = firstbithigh(_e138);
    int _e140 = i;
    i = asint(countbits(asuint(_e140)));
    int2 _e142 = i2_;
    i2_ = asint(countbits(asuint(_e142)));
    int3 _e144 = i3_;
    i3_ = asint(countbits(asuint(_e144)));
    int4 _e146 = i4_;
    i4_ = asint(countbits(asuint(_e146)));
    uint _e148 = u;
    u = countbits(_e148);
    uint2 _e150 = u2_;
    u2_ = countbits(_e150);
    uint3 _e152 = u3_;
    u3_ = countbits(_e152);
    uint4 _e154 = u4_;
    u4_ = countbits(_e154);
    int _e156 = i;
    i = asint(reversebits(asuint(_e156)));
    int2 _e158 = i2_;
    i2_ = asint(reversebits(asuint(_e158)));
    int3 _e160 = i3_;
    i3_ = asint(reversebits(asuint(_e160)));
    int4 _e162 = i4_;
    i4_ = asint(reversebits(asuint(_e162)));
    uint _e164 = u;
    u = reversebits(_e164);
    uint2 _e166 = u2_;
    u2_ = reversebits(_e166);
    uint3 _e168 = u3_;
    u3_ = reversebits(_e168);
    uint4 _e170 = u4_;
    u4_ = reversebits(_e170);
    return;
}

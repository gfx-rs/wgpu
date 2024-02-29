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

    float4 _expr28 = f4_;
    u = uint((int(round(clamp(_expr28[0], -1.0, 1.0) * 127.0)) & 0xFF) | ((int(round(clamp(_expr28[1], -1.0, 1.0) * 127.0)) & 0xFF) << 8) | ((int(round(clamp(_expr28[2], -1.0, 1.0) * 127.0)) & 0xFF) << 16) | ((int(round(clamp(_expr28[3], -1.0, 1.0) * 127.0)) & 0xFF) << 24));
    float4 _expr30 = f4_;
    u = (uint(round(clamp(_expr30[0], 0.0, 1.0) * 255.0)) | uint(round(clamp(_expr30[1], 0.0, 1.0) * 255.0)) << 8 | uint(round(clamp(_expr30[2], 0.0, 1.0) * 255.0)) << 16 | uint(round(clamp(_expr30[3], 0.0, 1.0) * 255.0)) << 24);
    float2 _expr32 = f2_;
    u = uint((int(round(clamp(_expr32[0], -1.0, 1.0) * 32767.0)) & 0xFFFF) | ((int(round(clamp(_expr32[1], -1.0, 1.0) * 32767.0)) & 0xFFFF) << 16));
    float2 _expr34 = f2_;
    u = (uint(round(clamp(_expr34[0], 0.0, 1.0) * 65535.0)) | uint(round(clamp(_expr34[1], 0.0, 1.0) * 65535.0)) << 16);
    float2 _expr36 = f2_;
    u = (f32tof16(_expr36[0]) | f32tof16(_expr36[1]) << 16);
    uint _expr38 = u;
    f4_ = (float4(int4(_expr38 << 24, _expr38 << 16, _expr38 << 8, _expr38) >> 24) / 127.0);
    uint _expr40 = u;
    f4_ = (float4(_expr40 & 0xFF, _expr40 >> 8 & 0xFF, _expr40 >> 16 & 0xFF, _expr40 >> 24) / 255.0);
    uint _expr42 = u;
    f2_ = (float2(int2(_expr42 << 16, _expr42) >> 16) / 32767.0);
    uint _expr44 = u;
    f2_ = (float2(_expr44 & 0xFFFF, _expr44 >> 16) / 65535.0);
    uint _expr46 = u;
    f2_ = float2(f16tof32(_expr46), f16tof32((_expr46) >> 16));
    int _expr48 = i;
    int _expr49 = i;
    i = naga_insertBits(_expr48, _expr49, 5u, 10u);
    int2 _expr53 = i2_;
    int2 _expr54 = i2_;
    i2_ = naga_insertBits(_expr53, _expr54, 5u, 10u);
    int3 _expr58 = i3_;
    int3 _expr59 = i3_;
    i3_ = naga_insertBits(_expr58, _expr59, 5u, 10u);
    int4 _expr63 = i4_;
    int4 _expr64 = i4_;
    i4_ = naga_insertBits(_expr63, _expr64, 5u, 10u);
    uint _expr68 = u;
    uint _expr69 = u;
    u = naga_insertBits(_expr68, _expr69, 5u, 10u);
    uint2 _expr73 = u2_;
    uint2 _expr74 = u2_;
    u2_ = naga_insertBits(_expr73, _expr74, 5u, 10u);
    uint3 _expr78 = u3_;
    uint3 _expr79 = u3_;
    u3_ = naga_insertBits(_expr78, _expr79, 5u, 10u);
    uint4 _expr83 = u4_;
    uint4 _expr84 = u4_;
    u4_ = naga_insertBits(_expr83, _expr84, 5u, 10u);
    int _expr88 = i;
    i = naga_extractBits(_expr88, 5u, 10u);
    int2 _expr92 = i2_;
    i2_ = naga_extractBits(_expr92, 5u, 10u);
    int3 _expr96 = i3_;
    i3_ = naga_extractBits(_expr96, 5u, 10u);
    int4 _expr100 = i4_;
    i4_ = naga_extractBits(_expr100, 5u, 10u);
    uint _expr104 = u;
    u = naga_extractBits(_expr104, 5u, 10u);
    uint2 _expr108 = u2_;
    u2_ = naga_extractBits(_expr108, 5u, 10u);
    uint3 _expr112 = u3_;
    u3_ = naga_extractBits(_expr112, 5u, 10u);
    uint4 _expr116 = u4_;
    u4_ = naga_extractBits(_expr116, 5u, 10u);
    int _expr120 = i;
    i = asint(firstbitlow(_expr120));
    uint2 _expr122 = u2_;
    u2_ = firstbitlow(_expr122);
    int3 _expr124 = i3_;
    i3_ = asint(firstbithigh(_expr124));
    uint3 _expr126 = u3_;
    u3_ = firstbithigh(_expr126);
    int _expr128 = i;
    i = asint(firstbithigh(_expr128));
    uint _expr130 = u;
    u = firstbithigh(_expr130);
    int _expr132 = i;
    i = asint(countbits(asuint(_expr132)));
    int2 _expr134 = i2_;
    i2_ = asint(countbits(asuint(_expr134)));
    int3 _expr136 = i3_;
    i3_ = asint(countbits(asuint(_expr136)));
    int4 _expr138 = i4_;
    i4_ = asint(countbits(asuint(_expr138)));
    uint _expr140 = u;
    u = countbits(_expr140);
    uint2 _expr142 = u2_;
    u2_ = countbits(_expr142);
    uint3 _expr144 = u3_;
    u3_ = countbits(_expr144);
    uint4 _expr146 = u4_;
    u4_ = countbits(_expr146);
    int _expr148 = i;
    i = asint(reversebits(asuint(_expr148)));
    int2 _expr150 = i2_;
    i2_ = asint(reversebits(asuint(_expr150)));
    int3 _expr152 = i3_;
    i3_ = asint(reversebits(asuint(_expr152)));
    int4 _expr154 = i4_;
    i4_ = asint(reversebits(asuint(_expr154)));
    uint _expr156 = u;
    u = reversebits(_expr156);
    uint2 _expr158 = u2_;
    u2_ = reversebits(_expr158);
    uint3 _expr160 = u3_;
    u3_ = reversebits(_expr160);
    uint4 _expr162 = u4_;
    u4_ = reversebits(_expr162);
    return;
}

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
    int4 _expr38 = i4_;
    u = uint((_expr38[0] & 0xFF) | ((_expr38[1] & 0xFF) << 8) | ((_expr38[2] & 0xFF) << 16) | ((_expr38[3] & 0xFF) << 24));
    uint4 _expr40 = u4_;
    u = (_expr40[0] & 0xFF) | ((_expr40[1] & 0xFF) << 8) | ((_expr40[2] & 0xFF) << 16) | ((_expr40[3] & 0xFF) << 24);
    uint _expr42 = u;
    f4_ = (float4(int4(_expr42 << 24, _expr42 << 16, _expr42 << 8, _expr42) >> 24) / 127.0);
    uint _expr44 = u;
    f4_ = (float4(_expr44 & 0xFF, _expr44 >> 8 & 0xFF, _expr44 >> 16 & 0xFF, _expr44 >> 24) / 255.0);
    uint _expr46 = u;
    f2_ = (float2(int2(_expr46 << 16, _expr46) >> 16) / 32767.0);
    uint _expr48 = u;
    f2_ = (float2(_expr48 & 0xFFFF, _expr48 >> 16) / 65535.0);
    uint _expr50 = u;
    f2_ = float2(f16tof32(_expr50), f16tof32((_expr50) >> 16));
    uint _expr52 = u;
    i4_ = int4(_expr52, _expr52 >> 8, _expr52 >> 16, _expr52 >> 24) << 24 >> 24;
    uint _expr54 = u;
    u4_ = uint4(_expr54, _expr54 >> 8, _expr54 >> 16, _expr54 >> 24) << 24 >> 24;
    int _expr56 = i;
    int _expr57 = i;
    i = naga_insertBits(_expr56, _expr57, 5u, 10u);
    int2 _expr61 = i2_;
    int2 _expr62 = i2_;
    i2_ = naga_insertBits(_expr61, _expr62, 5u, 10u);
    int3 _expr66 = i3_;
    int3 _expr67 = i3_;
    i3_ = naga_insertBits(_expr66, _expr67, 5u, 10u);
    int4 _expr71 = i4_;
    int4 _expr72 = i4_;
    i4_ = naga_insertBits(_expr71, _expr72, 5u, 10u);
    uint _expr76 = u;
    uint _expr77 = u;
    u = naga_insertBits(_expr76, _expr77, 5u, 10u);
    uint2 _expr81 = u2_;
    uint2 _expr82 = u2_;
    u2_ = naga_insertBits(_expr81, _expr82, 5u, 10u);
    uint3 _expr86 = u3_;
    uint3 _expr87 = u3_;
    u3_ = naga_insertBits(_expr86, _expr87, 5u, 10u);
    uint4 _expr91 = u4_;
    uint4 _expr92 = u4_;
    u4_ = naga_insertBits(_expr91, _expr92, 5u, 10u);
    int _expr96 = i;
    i = naga_extractBits(_expr96, 5u, 10u);
    int2 _expr100 = i2_;
    i2_ = naga_extractBits(_expr100, 5u, 10u);
    int3 _expr104 = i3_;
    i3_ = naga_extractBits(_expr104, 5u, 10u);
    int4 _expr108 = i4_;
    i4_ = naga_extractBits(_expr108, 5u, 10u);
    uint _expr112 = u;
    u = naga_extractBits(_expr112, 5u, 10u);
    uint2 _expr116 = u2_;
    u2_ = naga_extractBits(_expr116, 5u, 10u);
    uint3 _expr120 = u3_;
    u3_ = naga_extractBits(_expr120, 5u, 10u);
    uint4 _expr124 = u4_;
    u4_ = naga_extractBits(_expr124, 5u, 10u);
    int _expr128 = i;
    i = asint(firstbitlow(_expr128));
    uint2 _expr130 = u2_;
    u2_ = firstbitlow(_expr130);
    int3 _expr132 = i3_;
    i3_ = asint(firstbithigh(_expr132));
    uint3 _expr134 = u3_;
    u3_ = firstbithigh(_expr134);
    int _expr136 = i;
    i = asint(firstbithigh(_expr136));
    uint _expr138 = u;
    u = firstbithigh(_expr138);
    int _expr140 = i;
    i = asint(countbits(asuint(_expr140)));
    int2 _expr142 = i2_;
    i2_ = asint(countbits(asuint(_expr142)));
    int3 _expr144 = i3_;
    i3_ = asint(countbits(asuint(_expr144)));
    int4 _expr146 = i4_;
    i4_ = asint(countbits(asuint(_expr146)));
    uint _expr148 = u;
    u = countbits(_expr148);
    uint2 _expr150 = u2_;
    u2_ = countbits(_expr150);
    uint3 _expr152 = u3_;
    u3_ = countbits(_expr152);
    uint4 _expr154 = u4_;
    u4_ = countbits(_expr154);
    int _expr156 = i;
    i = asint(reversebits(asuint(_expr156)));
    int2 _expr158 = i2_;
    i2_ = asint(reversebits(asuint(_expr158)));
    int3 _expr160 = i3_;
    i3_ = asint(reversebits(asuint(_expr160)));
    int4 _expr162 = i4_;
    i4_ = asint(reversebits(asuint(_expr162)));
    uint _expr164 = u;
    u = reversebits(_expr164);
    uint2 _expr166 = u2_;
    u2_ = reversebits(_expr166);
    uint3 _expr168 = u3_;
    u3_ = reversebits(_expr168);
    uint4 _expr170 = u4_;
    u4_ = reversebits(_expr170);
    return;
}

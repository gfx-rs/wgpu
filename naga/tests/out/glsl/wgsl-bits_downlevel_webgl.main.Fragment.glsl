#version 300 es

precision highp float;
precision highp int;


uint naga_count_one_bits(uint e) {
    uint v = e - ((e >> 1u) & 0x55555555u);
    v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);
    v = (v + (v >> 4u)) & 0x0F0F0F0Fu;
    return (v * 0x01010101u) >> 24u;
}

uvec2 naga_count_one_bits(uvec2 e) {
    return uvec2(naga_count_one_bits(e.x), naga_count_one_bits(e.y));
}

uvec3 naga_count_one_bits(uvec3 e) {
    return uvec3(naga_count_one_bits(e.x), naga_count_one_bits(e.y), naga_count_one_bits(e.z));
}

uvec4 naga_count_one_bits(uvec4 e) {
    return uvec4(naga_count_one_bits(e.x), naga_count_one_bits(e.y), naga_count_one_bits(e.z), naga_count_one_bits(e.w));
}

int naga_count_one_bits(int e) {
    return int(naga_count_one_bits(uint(e)));
}

ivec2 naga_count_one_bits(ivec2 e) {
    return ivec2(naga_count_one_bits(e.x), naga_count_one_bits(e.y));
}

ivec3 naga_count_one_bits(ivec3 e) {
    return ivec3(naga_count_one_bits(e.x), naga_count_one_bits(e.y), naga_count_one_bits(e.z));
}

ivec4 naga_count_one_bits(ivec4 e) {
    return ivec4(naga_count_one_bits(e.x), naga_count_one_bits(e.y), naga_count_one_bits(e.z), naga_count_one_bits(e.w));
}

uint naga_reverse_bits(uint e) {
    uint v = e;
    v = ((v >> 1u) & 0x55555555u) | ((v & 0x55555555u) << 1u);
    v = ((v >> 2u) & 0x33333333u) | ((v & 0x33333333u) << 2u);
    v = ((v >> 4u) & 0x0F0F0F0Fu) | ((v & 0x0F0F0F0Fu) << 4u);
    v = ((v >> 8u) & 0x00FF00FFu) | ((v & 0x00FF00FFu) << 8u);
    return (v >> 16u) | (v << 16u);
}

uvec2 naga_reverse_bits(uvec2 e) {
    return uvec2(naga_reverse_bits(e.x), naga_reverse_bits(e.y));
}

uvec3 naga_reverse_bits(uvec3 e) {
    return uvec3(naga_reverse_bits(e.x), naga_reverse_bits(e.y), naga_reverse_bits(e.z));
}

uvec4 naga_reverse_bits(uvec4 e) {
    return uvec4(naga_reverse_bits(e.x), naga_reverse_bits(e.y), naga_reverse_bits(e.z), naga_reverse_bits(e.w));
}

int naga_reverse_bits(int e) {
    return int(naga_reverse_bits(uint(e)));
}

ivec2 naga_reverse_bits(ivec2 e) {
    return ivec2(naga_reverse_bits(e.x), naga_reverse_bits(e.y));
}

ivec3 naga_reverse_bits(ivec3 e) {
    return ivec3(naga_reverse_bits(e.x), naga_reverse_bits(e.y), naga_reverse_bits(e.z));
}

ivec4 naga_reverse_bits(ivec4 e) {
    return ivec4(naga_reverse_bits(e.x), naga_reverse_bits(e.y), naga_reverse_bits(e.z), naga_reverse_bits(e.w));
}

uint naga_extract_bits(uint e, uint offset, uint count) {
    uint o = min(offset, 32u);
    uint c = min(count, 32u - o);
    if (c == 0u) {
        return 0u;
    }
    if (c == 32u) {
        return e;
    }
    return (e >> o) & ((1u << c) - 1u);
}

uvec2 naga_extract_bits(uvec2 e, uint offset, uint count) {
    return uvec2(naga_extract_bits(e.x, offset, count), naga_extract_bits(e.y, offset, count));
}

uvec3 naga_extract_bits(uvec3 e, uint offset, uint count) {
    return uvec3(naga_extract_bits(e.x, offset, count), naga_extract_bits(e.y, offset, count), naga_extract_bits(e.z, offset, count));
}

uvec4 naga_extract_bits(uvec4 e, uint offset, uint count) {
    return uvec4(naga_extract_bits(e.x, offset, count), naga_extract_bits(e.y, offset, count), naga_extract_bits(e.z, offset, count), naga_extract_bits(e.w, offset, count));
}

int naga_extract_bits(int e, uint offset, uint count) {
    uint o = min(offset, 32u);
    uint c = min(count, 32u - o);
    if (c == 0u) {
        return 0;
    }
    if (c == 32u) {
        return e;
    }
    return int(uint(e) << (32u - o - c)) >> (32u - c);
}

ivec2 naga_extract_bits(ivec2 e, uint offset, uint count) {
    return ivec2(naga_extract_bits(e.x, offset, count), naga_extract_bits(e.y, offset, count));
}

ivec3 naga_extract_bits(ivec3 e, uint offset, uint count) {
    return ivec3(naga_extract_bits(e.x, offset, count), naga_extract_bits(e.y, offset, count), naga_extract_bits(e.z, offset, count));
}

ivec4 naga_extract_bits(ivec4 e, uint offset, uint count) {
    return ivec4(naga_extract_bits(e.x, offset, count), naga_extract_bits(e.y, offset, count), naga_extract_bits(e.z, offset, count), naga_extract_bits(e.w, offset, count));
}

uint naga_insert_bits(uint e, uint newbits, uint offset, uint count) {
    uint o = min(offset, 32u);
    uint c = min(count, 32u - o);
    if (c == 0u) {
        return e;
    }
    if (c == 32u) {
        return newbits;
    }
    uint mask = ((1u << c) - 1u) << o;
    return (e & ~mask) | ((newbits << o) & mask);
}

uvec2 naga_insert_bits(uvec2 e, uvec2 newbits, uint offset, uint count) {
    return uvec2(naga_insert_bits(e.x, newbits.x, offset, count), naga_insert_bits(e.y, newbits.y, offset, count));
}

uvec3 naga_insert_bits(uvec3 e, uvec3 newbits, uint offset, uint count) {
    return uvec3(naga_insert_bits(e.x, newbits.x, offset, count), naga_insert_bits(e.y, newbits.y, offset, count), naga_insert_bits(e.z, newbits.z, offset, count));
}

uvec4 naga_insert_bits(uvec4 e, uvec4 newbits, uint offset, uint count) {
    return uvec4(naga_insert_bits(e.x, newbits.x, offset, count), naga_insert_bits(e.y, newbits.y, offset, count), naga_insert_bits(e.z, newbits.z, offset, count), naga_insert_bits(e.w, newbits.w, offset, count));
}

int naga_insert_bits(int e, int newbits, uint offset, uint count) {
    return int(naga_insert_bits(uint(e), uint(newbits), offset, count));
}

ivec2 naga_insert_bits(ivec2 e, ivec2 newbits, uint offset, uint count) {
    return ivec2(naga_insert_bits(e.x, newbits.x, offset, count), naga_insert_bits(e.y, newbits.y, offset, count));
}

ivec3 naga_insert_bits(ivec3 e, ivec3 newbits, uint offset, uint count) {
    return ivec3(naga_insert_bits(e.x, newbits.x, offset, count), naga_insert_bits(e.y, newbits.y, offset, count), naga_insert_bits(e.z, newbits.z, offset, count));
}

ivec4 naga_insert_bits(ivec4 e, ivec4 newbits, uint offset, uint count) {
    return ivec4(naga_insert_bits(e.x, newbits.x, offset, count), naga_insert_bits(e.y, newbits.y, offset, count), naga_insert_bits(e.z, newbits.z, offset, count), naga_insert_bits(e.w, newbits.w, offset, count));
}

uint naga_count_trailing_zeros(uint e) {
    if (e == 0u) {
        return 32u;
    }
    uint v = e;
    uint n = 0u;
    if ((v & 0x0000FFFFu) == 0u) { n += 16u; v >>= 16u; }
    if ((v & 0x000000FFu) == 0u) { n += 8u; v >>= 8u; }
    if ((v & 0x0000000Fu) == 0u) { n += 4u; v >>= 4u; }
    if ((v & 0x00000003u) == 0u) { n += 2u; v >>= 2u; }
    if ((v & 0x00000001u) == 0u) { n += 1u; }
    return n;
}

uvec3 naga_count_trailing_zeros(uvec3 e) {
    return uvec3(naga_count_trailing_zeros(e.x), naga_count_trailing_zeros(e.y), naga_count_trailing_zeros(e.z));
}

int naga_count_trailing_zeros(int e) {
    return int(naga_count_trailing_zeros(uint(e)));
}

ivec2 naga_count_trailing_zeros(ivec2 e) {
    return ivec2(naga_count_trailing_zeros(e.x), naga_count_trailing_zeros(e.y));
}

uint naga_first_trailing_bit(uint e) {
    return e == 0u ? 0xFFFFFFFFu : naga_count_trailing_zeros(e);
}

uvec2 naga_first_trailing_bit(uvec2 e) {
    return uvec2(naga_first_trailing_bit(e.x), naga_first_trailing_bit(e.y));
}

int naga_first_trailing_bit(int e) {
    return int(naga_first_trailing_bit(uint(e)));
}

uint naga_count_leading_zeros(uint e) {
    if (e == 0u) {
        return 32u;
    }
    uint v = e;
    uint n = 0u;
    if ((v & 0xFFFF0000u) == 0u) { n += 16u; v <<= 16u; }
    if ((v & 0xFF000000u) == 0u) { n += 8u; v <<= 8u; }
    if ((v & 0xF0000000u) == 0u) { n += 4u; v <<= 4u; }
    if ((v & 0xC0000000u) == 0u) { n += 2u; v <<= 2u; }
    if ((v & 0x80000000u) == 0u) { n += 1u; }
    return n;
}

uvec2 naga_count_leading_zeros(uvec2 e) {
    return uvec2(naga_count_leading_zeros(e.x), naga_count_leading_zeros(e.y));
}

int naga_count_leading_zeros(int e) {
    return int(naga_count_leading_zeros(uint(e)));
}

ivec4 naga_count_leading_zeros(ivec4 e) {
    return ivec4(naga_count_leading_zeros(e.x), naga_count_leading_zeros(e.y), naga_count_leading_zeros(e.z), naga_count_leading_zeros(e.w));
}

uint naga_first_leading_bit(uint e) {
    return e == 0u ? 0xFFFFFFFFu : 31u - naga_count_leading_zeros(e);
}

uvec3 naga_first_leading_bit(uvec3 e) {
    return uvec3(naga_first_leading_bit(e.x), naga_first_leading_bit(e.y), naga_first_leading_bit(e.z));
}

int naga_first_leading_bit(int e) {
    return int(naga_first_leading_bit(uint(e < 0 ? ~e : e)));
}

ivec3 naga_first_leading_bit(ivec3 e) {
    return ivec3(naga_first_leading_bit(e.x), naga_first_leading_bit(e.y), naga_first_leading_bit(e.z));
}

void main() {
    int i = 0;
    ivec2 i2_ = ivec2(0);
    ivec3 i3_ = ivec3(0);
    ivec4 i4_ = ivec4(0);
    uint u = 0u;
    uvec2 u2_ = uvec2(0u);
    uvec3 u3_ = uvec3(0u);
    uvec4 u4_ = uvec4(0u);
    vec2 f2_ = vec2(0.0);
    vec4 f4_ = vec4(0.0);
    ivec4 _e28 = i4_;
    u = uint((_e28[0] & 0xFF) | ((_e28[1] & 0xFF) << 8) | ((_e28[2] & 0xFF) << 16) | ((_e28[3] & 0xFF) << 24));
    uvec4 _e30 = u4_;
    u = (_e30[0] & 0xFFu) | ((_e30[1] & 0xFFu) << 8) | ((_e30[2] & 0xFFu) << 16) | ((_e30[3] & 0xFFu) << 24);
    uint _e32 = u;
    f4_ = (vec4(ivec4(_e32 << 24, _e32 << 16, _e32 << 8, _e32) >> 24) / 127.0);
    uint _e34 = u;
    f4_ = (vec4(_e34 & 0xFFu, _e34 >> 8 & 0xFFu, _e34 >> 16 & 0xFFu, _e34 >> 24) / 255.0);
    uint _e36 = u;
    f2_ = unpackSnorm2x16(_e36);
    uint _e38 = u;
    f2_ = unpackUnorm2x16(_e38);
    int _e40 = i;
    int _e41 = i;
    i = naga_insert_bits(_e40, _e41, 5u, 10u);
    ivec2 _e45 = i2_;
    ivec2 _e46 = i2_;
    i2_ = naga_insert_bits(_e45, _e46, 5u, 10u);
    ivec3 _e50 = i3_;
    ivec3 _e51 = i3_;
    i3_ = naga_insert_bits(_e50, _e51, 5u, 10u);
    ivec4 _e55 = i4_;
    ivec4 _e56 = i4_;
    i4_ = naga_insert_bits(_e55, _e56, 5u, 10u);
    uint _e60 = u;
    uint _e61 = u;
    u = naga_insert_bits(_e60, _e61, 5u, 10u);
    uvec2 _e65 = u2_;
    uvec2 _e66 = u2_;
    u2_ = naga_insert_bits(_e65, _e66, 5u, 10u);
    uvec3 _e70 = u3_;
    uvec3 _e71 = u3_;
    u3_ = naga_insert_bits(_e70, _e71, 5u, 10u);
    uvec4 _e75 = u4_;
    uvec4 _e76 = u4_;
    u4_ = naga_insert_bits(_e75, _e76, 5u, 10u);
    int _e80 = i;
    i = naga_extract_bits(_e80, 5u, 10u);
    ivec2 _e84 = i2_;
    i2_ = naga_extract_bits(_e84, 5u, 10u);
    ivec3 _e88 = i3_;
    i3_ = naga_extract_bits(_e88, 5u, 10u);
    ivec4 _e92 = i4_;
    i4_ = naga_extract_bits(_e92, 5u, 10u);
    uint _e96 = u;
    u = naga_extract_bits(_e96, 5u, 10u);
    uvec2 _e100 = u2_;
    u2_ = naga_extract_bits(_e100, 5u, 10u);
    uvec3 _e104 = u3_;
    u3_ = naga_extract_bits(_e104, 5u, 10u);
    uvec4 _e108 = u4_;
    u4_ = naga_extract_bits(_e108, 5u, 10u);
    int _e112 = i;
    i = naga_first_trailing_bit(_e112);
    uvec2 _e114 = u2_;
    u2_ = naga_first_trailing_bit(_e114);
    ivec3 _e116 = i3_;
    i3_ = naga_first_leading_bit(_e116);
    uvec3 _e118 = u3_;
    u3_ = naga_first_leading_bit(_e118);
    int _e120 = i;
    i = naga_first_leading_bit(_e120);
    uint _e122 = u;
    u = naga_first_leading_bit(_e122);
    int _e124 = i;
    i = naga_count_one_bits(_e124);
    ivec2 _e126 = i2_;
    i2_ = naga_count_one_bits(_e126);
    ivec3 _e128 = i3_;
    i3_ = naga_count_one_bits(_e128);
    ivec4 _e130 = i4_;
    i4_ = naga_count_one_bits(_e130);
    uint _e132 = u;
    u = naga_count_one_bits(_e132);
    uvec2 _e134 = u2_;
    u2_ = naga_count_one_bits(_e134);
    uvec3 _e136 = u3_;
    u3_ = naga_count_one_bits(_e136);
    uvec4 _e138 = u4_;
    u4_ = naga_count_one_bits(_e138);
    int _e140 = i;
    i = naga_reverse_bits(_e140);
    ivec2 _e142 = i2_;
    i2_ = naga_reverse_bits(_e142);
    ivec3 _e144 = i3_;
    i3_ = naga_reverse_bits(_e144);
    ivec4 _e146 = i4_;
    i4_ = naga_reverse_bits(_e146);
    uint _e148 = u;
    u = naga_reverse_bits(_e148);
    uvec2 _e150 = u2_;
    u2_ = naga_reverse_bits(_e150);
    uvec3 _e152 = u3_;
    u3_ = naga_reverse_bits(_e152);
    uvec4 _e154 = u4_;
    u4_ = naga_reverse_bits(_e154);
    int _e156 = i;
    i = naga_count_trailing_zeros(_e156);
    ivec2 _e158 = i2_;
    i2_ = naga_count_trailing_zeros(_e158);
    uint _e160 = u;
    u = naga_count_trailing_zeros(_e160);
    uvec3 _e162 = u3_;
    u3_ = naga_count_trailing_zeros(_e162);
    int _e164 = i;
    i = naga_count_leading_zeros(_e164);
    ivec4 _e166 = i4_;
    i4_ = naga_count_leading_zeros(_e166);
    uint _e168 = u;
    u = naga_count_leading_zeros(_e168);
    uvec2 _e170 = u2_;
    u2_ = naga_count_leading_zeros(_e170);
    return;
}


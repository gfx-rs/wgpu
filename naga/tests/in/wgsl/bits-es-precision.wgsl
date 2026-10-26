// GLSL ES declares `bitCount`, `findLSB` and `findMSB` as returning `lowp int`.
// Conversions of and arithmetic on their results must not inherit that precision.

@compute @workgroup_size(1)
fn main() {
    var u = 0xdeadbeefu;
    var i = -2;
    var f = 0.0;
    var u2 = vec2<u32>(0xdeadbeefu, 0u);
    u = ~countOneBits(u);
    u = countOneBits(u) * 0x10001u;
    u = ~countTrailingZeros(u);
    u = ~firstTrailingBit(u);
    u = ~countLeadingZeros(u);
    u = ~firstLeadingBit(u);
    i = countOneBits(i) - 100000;
    i = ~countTrailingZeros(i);
    i = ~firstTrailingBit(i);
    i = ~firstLeadingBit(i);
    i = ~countLeadingZeros(i);
    f = f32(firstTrailingBit(u));
    f = f32(firstLeadingBit(u));
    u2 = ~countOneBits(u2) * vec2<u32>(0x10001u);
}

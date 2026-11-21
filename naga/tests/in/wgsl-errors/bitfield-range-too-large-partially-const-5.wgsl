// `offset` and `count` are const expressions here too, just not literals.

const OFFSET = 30u;
         fn f(e: u32) -> u32 { return extractBits(e, OFFSET, 8u); }

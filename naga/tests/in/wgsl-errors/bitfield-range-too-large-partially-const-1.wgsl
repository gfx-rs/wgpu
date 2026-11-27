// The operand is a function parameter, so these calls never reach constant
// evaluation at all.

fn f(e: u32) -> u32 { return extractBits(e, 30u, 8u); }

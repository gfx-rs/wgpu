// offset + count  past the width of the operand is an error w shader creation
// when both are const expressions.

const x = insertBits(vec2(0u), vec2(0u), 4294967295u, 4294967295u);

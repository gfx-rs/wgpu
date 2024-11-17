// Sourced from https://www.w3.org/TR/WGSL/#const-assert-statement
const x = 1;
const y = 2;
const_assert x < y; // valid at module-scope.
const_assert(y != 0); // parentheses are optional.

fn foo() {
  const z = x + y - 2;
  const_assert z > 0; // valid in functions.
  const_assert(z > 0);
}
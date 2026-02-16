// Sourced from https://www.w3.org/TR/WGSL/#const-assert-statement and the CTS.
const x = 1;
const y = 2;
const_assert x < y; // valid at module-scope.
const_assert(y != 0); // parentheses are optional.

// Ensure abstract-typed consts can be compared to different concrete types
const_assert x == 1i;
const_assert x > 0u;
const_assert x < 2.0f;

const g_false = false;
const_assert (!((g_false) || (any(vec3(false, false, false)))));

@compute @workgroup_size(1)
fn foo() {
  const z = x + y - 2;
  const l_false = false;
  const_assert z > 0; // valid in functions.
  const_assert(z > 0);

  const_assert z == 1i;
  const_assert z > 0u;
  const_assert z < 2.0f;

  const_assert (!((g_false) || (any(vec3(false, false, false)))));
  const_assert (!((l_false) || (any(vec3(false, false, false)))));
}

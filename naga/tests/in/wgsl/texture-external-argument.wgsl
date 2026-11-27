@group(0) @binding(0)
var tex: texture_external;

fn inner(t: texture_external) -> u32 {
  return 0u;
}

fn outer(t: texture_external) -> u32 {
  return inner(t);
}

@compute @workgroup_size(1)
fn main() {
  _ = outer(tex);
}

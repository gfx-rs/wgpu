// NOTE: This allows us to suppress compaction below, to force the handling of identifiers
// containing Unicode.
@group(0) @binding(0)
var<storage> asdf: f32;

fn compute() -> f32 {
  let θ2 = asdf + 9001.0;
  return θ2;
}

@compute @workgroup_size(1, 1)
fn main() {
  compute();
}

// There are similar validator tests in `validation.rs`.
// Rejected in constructor lowering

struct Unsized { data: array<f32> }
fn main() {
    var not_okay: Unsized = Unsized();
}

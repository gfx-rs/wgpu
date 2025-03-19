// Conversion of initializer expressions
// (the abstract-types-* tests have a bunch more like this)
const ic0: vec2f = vec2(1, 1) + vec2(1.0, 1.0);

// Conversion by value constructors
//let vc0 = i32(1.0);  // https://github.com/gfx-rs/wgpu/issues/7312
// etc. (also create the locals versions below)

@compute @workgroup_size(1)
fn main() {
    const ic0: vec2f = vec2(1, 1) + vec2(1.0, 1.0);

    let lc0: vec2f = vec2(1, 1) + vec2(1.0, 1.0);

    var vc0: vec2f = vec2(1, 1) + vec2(1.0, 1.0);
}

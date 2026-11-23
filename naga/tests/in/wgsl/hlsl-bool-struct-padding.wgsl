// Structs whose naga size is not a multiple of 4, nested in other structs.
// The HLSL padding must not need 16-bit types.

struct S {
    b: bool,
}

struct V {
    b: vec2<bool>,
}

struct Middle {
    s: S,
    x: f32,
    v: V,
    y: f32,
}

struct Last {
    x: f32,
    s: S,
}

struct ArrayMiddle {
    a: array<S, 3>,
    @align(16) x: f32,
}

var<private> middle: Middle;
var<private> last: Last;
var<private> array_middle: ArrayMiddle;

@group(0) @binding(0)
var<storage, read_write> o: f32;

@compute @workgroup_size(1)
fn main() {
    middle.s.b = o > 1.0;
    last.s.b = o > 2.0;
    array_middle.a[1].b = o > 3.0;
    var x = 0.0;
    if middle.s.b && middle.v.b.y && last.s.b && array_middle.a[1].b {
        x = middle.x + middle.y + last.x + array_middle.x;
    }
    o = x;
}

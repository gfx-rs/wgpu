let Foo_1: bool = true;

var<workgroup> wg: array<f32,10u>;
var<workgroup> at_1: atomic<u32>;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    var Foo: f32 = 1.0;
    var at: bool = true;

    wg[3] = 1.0;
    atomicStore((&at_1), 2u);
    return;
}

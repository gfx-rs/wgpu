let Foo: bool = true;

var<workgroup> wg: array<f32,10u>;
var<workgroup> at: atomic<u32>;

[[stage(compute), workgroup_size(1, 1, 1)]]
fn main() {
    wg[3] = 1.0;
    atomicStore((&at), 2u);
    return;
}

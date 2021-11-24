// Global variable & constant declarations

let Foo: bool = true;

var<workgroup> wg : array<f32, 10u>;
var<workgroup> at: atomic<u32>;

[[stage(compute), workgroup_size(1)]]
fn main() {
	wg[3] = 1.0;
	atomicStore(&at, 2u);

    // Valid, Foo and at is in function scope
    var Foo: f32 = 1.0;
    var at: bool = true;
}

var<private> sink: u32 = 0;

fn simple() {
    var a = sink;
    let b = a;

    a = 2u;

    sink = b;
}

@compute @workgroup_size(1)
fn main() {
    simple();
}
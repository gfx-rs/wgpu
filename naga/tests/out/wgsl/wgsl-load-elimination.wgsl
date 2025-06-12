var<private> sink: u32 = 0u;

fn simple() {
    var a: u32;

    let _e1 = sink;
    a = _e1;
    let b = a;
    a = 2u;
    sink = b;
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    simple();
    return;
}

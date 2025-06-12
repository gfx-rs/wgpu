var<private> sink: u32 = 0u;

fn function() {
    var a: u32 = u32();

    let _e4 = sink;
    a = _e4;
    let _e5 = a;
    a = 2u;
    sink = _e5;
    return;
}

fn function_1() {
    function();
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    function_1();
}

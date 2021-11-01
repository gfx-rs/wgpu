[[block]]
struct type2 {
    member: i32;
};

[[group(0), binding(0)]]
var<storage, read_write> global: type2;

fn function1() {
    let e8: i32 = global.member;
    global.member = (e8 + 1);
    return;
}

[[stage(compute), workgroup_size(64, 1, 1)]]
fn main() {
    function1();
}

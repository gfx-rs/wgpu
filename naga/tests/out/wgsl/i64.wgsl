const k: u64 = 20lu;

var<private> v: i64 = 1i;

fn fi(x: i64) -> i64 {
    var z: i64;

    let y = (31i - 1002003004005006i);
    z = (y + 5i);
    return (((x + y) + 20i) + 50i);
}

fn fu(x_1: u64) -> u64 {
    var z_1: u64;

    let y_1 = (31lu + 1002003004005006lu);
    z_1 = (y_1 + 4lu);
    return (((x_1 + y_1) + k) + 34lu);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e1 = fu(67lu);
    let _e3 = fi(60i);
    return;
}

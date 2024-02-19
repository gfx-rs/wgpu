const k: u64 = 20lu;

var<private> v: i64 = 1li;

fn fi(x: i64) -> i64 {
    var z: i64;

    let y = (31li - 1002003004005006li);
    z = (y + 5li);
    return (((x + y) + 20li) + 50li);
}

fn fu(x_1: u64) -> u64 {
    var z_1: u64;

    let y_1 = (31lu + 1002003004005006lu);
    let v_1 = vec3<u64>(3lu, 4lu, 5lu);
    z_1 = (y_1 + 4lu);
    return ((((((x_1 + y_1) + k) + 34lu) + v_1.x) + v_1.y) + v_1.z);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e1 = fu(67lu);
    let _e3 = fi(60li);
    return;
}

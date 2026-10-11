alias A = array<u32, 5>;
alias B = array<u32, 5>;

struct Left {
    value: u32,
}

struct Right {
    value: u32,
}

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

fn first(indices: B) -> u32 {
    return indices[0];
}

fn return_array(indices: A) -> B {
    return indices;
}

@compute @workgroup_size(1)
fn main() {
    var a: A;
    a[0] = data[0];
    a[1] = data[1];
    a[2] = data[2];
    a[3] = data[3];
    a[4] = data[4];

    var b: B = a;
    var unnamed: array<u32, 5> = return_array(b);
    b = unnamed;

    var shorter: array<u32, 4>;
    var signed: array<i32, 5>;
    var left: array<Left, 5>;
    var right: array<Right, 5>;

    data[5] = first(a) + b[1] + shorter[0] + u32(signed[0])
        + left[0].value + right[0].value;
}

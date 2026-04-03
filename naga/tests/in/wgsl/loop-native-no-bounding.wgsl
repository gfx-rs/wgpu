// Tests for/while loop output with force_loop_bounding disabled (MSL, HLSL).
// The output should use native for/while without any bounding counter.

@group(0) @binding(0) var<storage, read_write> out: array<i32>;

fn simple_for() -> i32 {
    var sum = 0i;
    for (var i = 0i; i < 10i; i++) {
        sum += i;
    }
    return sum;
}

fn simple_while() -> i32 {
    var i = 0i;
    while i < 10i {
        i++;
    }
    return i;
}

fn for_with_continue() -> i32 {
    var sum = 0i;
    for (var i = 0i; i < 10i; i++) {
        if i == 5i {
            continue;
        }
        sum += i;
    }
    return sum;
}

fn nested_for_while() -> i32 {
    var sum = 0i;
    var i = 0i;
    while i < 3i {
        for (var j = 0i; j < 3i; j++) {
            sum += i * 3i + j;
        }
        i++;
    }
    return sum;
}

@compute @workgroup_size(1)
fn main() {
    out[0] = simple_for();
    out[1] = simple_while();
    out[2] = for_with_continue();
    out[3] = nested_for_while();
}

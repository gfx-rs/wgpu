// Tests for/while loop edge cases: native syntax paths and fallback paths.

@group(0) @binding(0) var<storage, read_write> out: array<i32>;

// Basic for loop — should emit native `for` on all text backends.
fn simple_for() -> i32 {
    var sum = 0i;
    for (var i = 0i; i < 10i; i++) {
        sum += i;
    }
    return sum;
}

// Basic while loop — should emit native `while` on all text backends.
fn simple_while() -> i32 {
    var i = 0i;
    while i < 10i {
        i++;
    }
    return i;
}

// For loop with `continue` — the update must still run.
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

// For loop with `break`.
fn for_with_break() -> i32 {
    var sum = 0i;
    for (var i = 0i; i < 10i; i++) {
        if i == 5i {
            break;
        }
        sum += i;
    }
    return sum;
}

// For loop with no condition — infinite until break.
fn for_infinite() -> i32 {
    var i = 0i;
    for (;;) {
        if i >= 10i {
            break;
        }
        i++;
    }
    return i;
}

// For loop with no update.
fn for_no_update() -> i32 {
    var i = 0i;
    for (; i < 10i;) {
        i++;
    }
    return i;
}

// While loop with break.
fn while_with_break() -> i32 {
    var i = 0i;
    while true {
        if i >= 10i {
            break;
        }
        i++;
    }
    return i;
}

// Nested for inside while.
fn nested_loops() -> i32 {
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

// For loop where the variable is NOT declared inside the loop header.
fn for_var_outside() -> i32 {
    var i = 0i;
    for (; i < 10i; i++) {
        // empty
    }
    return i;
}

@compute @workgroup_size(1)
fn main() {
    out[0] = simple_for();
    out[1] = simple_while();
    out[2] = for_with_continue();
    out[3] = for_with_break();
    out[4] = for_infinite();
    out[5] = for_no_update();
    out[6] = while_with_break();
    out[7] = nested_loops();
    out[8] = for_var_outside();
}

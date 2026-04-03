@group(0) @binding(0) 
var<storage, read_write> out: array<i32>;

fn simple_for() -> i32 {
    var sum: i32 = 0i;

    for (var i: i32 = 0i; (i < 10i); i = (i + 1i)) {
        let _e7 = sum;
        let _e8 = i;
        sum = (_e7 + _e8);
    }
    let _e13 = sum;
    return _e13;
}

fn simple_while() -> i32 {
    var i_1: i32 = 0i;

    while (i_1 < 10i) {
        let _e6 = i_1;
        i_1 = (_e6 + 1i);
    }
    let _e8 = i_1;
    return _e8;
}

fn for_with_continue() -> i32 {
    var sum_1: i32 = 0i;

    for (var i_2: i32 = 0i; (i_2 < 10i); i_2 = (i_2 + 1i)) {
        let _e7 = i_2;
        if (_e7 == 5i) {
            continue;
        }
        let _e10 = sum_1;
        let _e11 = i_2;
        sum_1 = (_e10 + _e11);
    }
    let _e16 = sum_1;
    return _e16;
}

fn for_with_break() -> i32 {
    var sum_2: i32 = 0i;

    for (var i_3: i32 = 0i; (i_3 < 10i); i_3 = (i_3 + 1i)) {
        let _e7 = i_3;
        if (_e7 == 5i) {
            break;
        }
        let _e10 = sum_2;
        let _e11 = i_3;
        sum_2 = (_e10 + _e11);
    }
    let _e16 = sum_2;
    return _e16;
}

fn for_infinite() -> i32 {
    var i_4: i32 = 0i;

    for (; ; ) {
        let _e2 = i_4;
        if (_e2 >= 10i) {
            break;
        }
        let _e6 = i_4;
        i_4 = (_e6 + 1i);
    }
    let _e8 = i_4;
    return _e8;
}

fn for_no_update() -> i32 {
    var i_5: i32 = 0i;

    for (; (i_5 < 10i); ) {
        let _e6 = i_5;
        i_5 = (_e6 + 1i);
    }
    let _e8 = i_5;
    return _e8;
}

fn while_with_break() -> i32 {
    var i_6: i32 = 0i;

    while true {
        let _e3 = i_6;
        if (_e3 >= 10i) {
            break;
        }
        let _e7 = i_6;
        i_6 = (_e7 + 1i);
    }
    let _e9 = i_6;
    return _e9;
}

fn nested_loops() -> i32 {
    var sum_3: i32 = 0i;
    var i_7: i32 = 0i;

    while (i_7 < 3i) {
        for (var j: i32 = 0i; (j < 3i); j = (j + 1i)) {
            let _e12 = sum_3;
            let _e13 = i_7;
            let _e16 = j;
            sum_3 = (_e12 + ((_e13 * 3i) + _e16));
        }
        let _e23 = i_7;
        i_7 = (_e23 + 1i);
    }
    let _e25 = sum_3;
    return _e25;
}

fn for_var_outside() -> i32 {
    var i_8: i32 = 0i;

    for (; (i_8 < 10i); i_8 = (i_8 + 1i)) {
    }
    let _e8 = i_8;
    return _e8;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e2 = simple_for();
    out[0] = _e2;
    let _e5 = simple_while();
    out[1] = _e5;
    let _e8 = for_with_continue();
    out[2] = _e8;
    let _e11 = for_with_break();
    out[3] = _e11;
    let _e14 = for_infinite();
    out[4] = _e14;
    let _e17 = for_no_update();
    out[5] = _e17;
    let _e20 = while_with_break();
    out[6] = _e20;
    let _e23 = nested_loops();
    out[7] = _e23;
    let _e26 = for_var_outside();
    out[8] = _e26;
    return;
}

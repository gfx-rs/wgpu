fn switchEmpty(a: i32) {
    var a_1: i32;

    a_1 = a;
    let _e2 = a_1;
    switch _e2 {
        default: {
        }
    }
    return;
}

fn switchNoDefault(a_2: i32) {
    var a_3: i32;

    a_3 = a_2;
    let _e2 = a_3;
    switch _e2 {
        case 0: {
        }
        default: {
        }
    }
    return;
}

fn switchNoLastBreak(a_4: i32) {
    var a_5: i32;
    var b: i32;

    a_5 = a_4;
    let _e2 = a_5;
    switch _e2 {
        default: {
            let _e3 = a_5;
            b = _e3;
        }
    }
    return;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

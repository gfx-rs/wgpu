fn breakIfEmpty() {
    loop {
        continuing {
            break if true;
        }
    }
    return;
}

fn breakIfEmptyBody(a: bool) {
    var b: bool;
    var c: bool;

    loop {
        continuing {
            b = a;
            let _e2 = b;
            c = (a != _e2);
            let _e5 = c;
            break if (a == _e5);
        }
    }
    return;
}

fn breakIf(a_1: bool) {
    var d: bool;
    var e: bool;

    loop {
        d = a_1;
        let _e2 = d;
        e = (a_1 != _e2);
        continuing {
            let _e5 = e;
            break if (a_1 == _e5);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    return;
}

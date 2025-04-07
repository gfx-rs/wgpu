fn main_1() {
    var i: i32 = 0i;

    loop {
        let _e2 = i;
        if !((_e2 < 1i)) {
            break;
        }
        {
        }
        continuing {
            let _e6 = i;
            i = (_e6 + 1i);
        }
    }
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

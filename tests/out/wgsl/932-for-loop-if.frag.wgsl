fn main_1() {
    var i: i32 = 0;

    loop {
        let _e2 = i;
        if !((_e2 < 1)) {
            break;
        }
        {
        }
        continuing {
            let _e6 = i;
            i = (_e6 + 1);
        }
    }
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

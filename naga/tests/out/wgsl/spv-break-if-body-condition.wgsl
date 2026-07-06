fn f_u0028_() {
    var i: i32;
    var acc: i32;

    i = 0i;
    acc = 0i;
    loop {
        let _e5 = i;
        let _e6 = acc;
        acc = (_e6 + _e5);
        continue;
        continuing {
            let _e9 = i;
            i = (_e9 + 1i);
            break if !((_e5 < 3i));
        }
    }
    return;
}

fn main_1() {
    f_u0028_();
    return;
}

@fragment 
fn main() {
    main_1();
}

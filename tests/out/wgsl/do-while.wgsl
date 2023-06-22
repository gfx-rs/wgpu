fn fb1_(cond: ptr<function, bool>) {
    loop {
        continue;
        continuing {
            let _e6 = (*cond);
            _ = !(_e6);
            break if !(_e6);
        }
    }
    return;
}

fn main_1() {
    var param: bool;

    param = false;
    fb1_((&param));
    return;
}

@fragment 
fn main() {
    main_1();
}

fn fb1_(cond: ptr<function, bool>) {
    loop {
        continue;
        continuing {
            let _e1 = (*cond);
            break if !(_e1);
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

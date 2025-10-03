fn f_u0028_b1_u003b(cond: ptr<function, bool>) {
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
    f_u0028_b1_u003b((&param));
    return;
}

@fragment 
fn main() {
    main_1();
}

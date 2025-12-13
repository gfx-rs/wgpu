fn main_1() {
    var x: i32 = 0i;
    var y: i32;
    var z: i32;

    loop {
        let _e2 = x;
        if !((_e2 < 10i)) {
            break;
        }
        {
            y = 0i;
            loop {
                let _e11 = y;
                if !((_e11 < 10i)) {
                    break;
                }
                {
                    z = 0i;
                    loop {
                        let _e20 = z;
                        if !((_e20 < 10i)) {
                            break;
                        }
                        {
                        }
                        continuing {
                            let _e24 = z;
                            z = (_e24 + 1i);
                        }
                    }
                }
                continuing {
                    let _e15 = y;
                    y = (_e15 + 1i);
                }
            }
        }
        continuing {
            let _e6 = x;
            x = (_e6 + 1i);
        }
    }
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

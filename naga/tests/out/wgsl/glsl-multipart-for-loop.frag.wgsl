fn main_1() {
    var a: f32 = 1f;
    var b: f32 = 0.25f;
    var i: i32 = 0i;

    loop {
        let _e6 = i;
        if !((_e6 < 25i)) {
            break;
        }
        {
            let _e13 = a;
            a = (_e13 - 0.02f);
        }
        continuing {
            let _e10 = b;
            b = (_e10 + 0.01f);
        }
    }
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

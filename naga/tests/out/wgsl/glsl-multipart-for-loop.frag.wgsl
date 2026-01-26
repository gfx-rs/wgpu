fn main_1() {
    var a: f32 = 1f;
    var b: f32 = 0.25f;
    var c: f32 = 1.5f;
    var i: i32 = 20i;

    i = 0i;
    let _e9 = c;
    c = (_e9 - 1f);
    loop {
        let _e12 = i;
        if !((_e12 < 25i)) {
            break;
        }
        {
            let _e22 = a;
            a = (_e22 - 0.02f);
        }
        continuing {
            let _e16 = i;
            i = (_e16 + 1i);
            let _e19 = b;
            b = (_e19 + 0.01f);
        }
    }
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

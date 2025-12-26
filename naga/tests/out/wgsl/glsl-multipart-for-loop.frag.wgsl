fn main_1() {
    var a: f32 = 1f;
    var b: f32 = 0.25f;
    var c: f32 = 1.5f;
    var i: i32 = 20i;

    let _e8 = c;
    c = (_e8 - 1f);
    loop {
        let _e11 = i;
        if !((_e11 < 25i)) {
            break;
        }
        {
            let _e18 = a;
            a = (_e18 - 0.02f);
        }
        continuing {
            let _e15 = b;
            b = (_e15 + 0.01f);
        }
    }
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

fn main1() {
    var i: i32 = 0;

    loop {
        let _e2: i32 = i;
        if (!((_e2 < 1))) {
            break;
        }
        {
        }
        continuing {
            let _e6: i32 = i;
            i = (_e6 + 1);
        }
    }
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}

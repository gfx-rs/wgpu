fn main1() {
    var i: i32 = 0;

    loop {
        let e2: i32 = i;
        if (!((e2 < 1))) {
            break;
        }
        {
        }
        continuing {
            let e6: i32 = i;
            i = (e6 + 1);
        }
    }
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}

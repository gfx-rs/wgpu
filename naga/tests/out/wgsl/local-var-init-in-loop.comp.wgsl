fn main_1() {
    var sum: vec4<f32> = vec4(0f);
    var i: i32 = 0i;
    var a: vec4<f32>;

    loop {
        let _e6 = i;
        if !((_e6 < 4i)) {
            break;
        }
        {
            a = vec4(1f);
            let _e17 = sum;
            let _e18 = a;
            sum = (_e17 + _e18);
        }
        continuing {
            let _e10 = i;
            i = (_e10 + 1i);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
    return;
}

fn main_1() {
    var sum: vec4<f32> = vec4(0.0);
    var i: i32 = 0;
    var a: vec4<f32> = vec4(1.0);

    loop {
        let _e6 = i;
        if !((_e6 < 4)) {
            break;
        }
        {
            let _e15 = vec4(1.0);
            a = _e15;
            let _e17 = sum;
            let _e18 = a;
            sum = (_e17 + _e18);
        }
        continuing {
            let _e10 = i;
            i = (_e10 + 1);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
    return;
}

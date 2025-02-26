fn main_1() {
    var sum: vec4<f32> = vec4(0f);
    var i: i32 = 0i;
    var a: vec4<f32>;

    loop {
        let _e5 = i;
        if !((_e5 < 4i)) {
            break;
        }
        {
            a = vec4(1f);
            let _e15 = sum;
            let _e16 = a;
            sum = (_e15 + _e16);
        }
        continuing {
            let _e9 = i;
            i = (_e9 + 1i);
        }
    }
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
    return;
}

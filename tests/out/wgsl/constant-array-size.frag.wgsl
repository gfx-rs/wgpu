struct Data {
    vecs: array<vec4<f32>, 42>,
}

const NUM_VECS: i32 = 42;

@group(1) @binding(0) 
var<uniform> global: Data;

fn function() -> vec4<f32> {
    var sum: vec4<f32>;
    var i: i32;

    sum = vec4(0.0);
    i = 0;
    loop {
        let _e9 = i;
        if !((_e9 < NUM_VECS)) {
            break;
        }
        {
            let _e15 = sum;
            let _e16 = i;
            let _e18 = global.vecs[_e16];
            sum = (_e15 + _e18);
        }
        continuing {
            let _e12 = i;
            i = (_e12 + 1);
        }
    }
    let _e20 = sum;
    return _e20;
}

fn main_1() {
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

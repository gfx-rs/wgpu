struct Data {
    vecs: array<vec4<f32>, 42>,
}

const NUM_VECS: i32 = 42i;

@group(1) @binding(0) 
var<uniform> global: Data;

fn function() -> vec4<f32> {
    var sum: vec4<f32> = vec4(0f);
    var i: i32 = 0i;

    loop {
        let _e8 = i;
        if !((_e8 < NUM_VECS)) {
            break;
        }
        {
            let _e14 = sum;
            let _e15 = i;
            let _e17 = global.vecs[_e15];
            sum = (_e14 + _e17);
        }
        continuing {
            let _e11 = i;
            i = (_e11 + 1i);
        }
    }
    let _e19 = sum;
    return _e19;
}

fn main_1() {
    let _e0 = function();
    return;
}

@fragment 
fn main() {
    main_1();
    return;
}

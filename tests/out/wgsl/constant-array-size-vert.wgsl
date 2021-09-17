[[block]]
struct Data {
    vecs: [[stride(16)]] array<vec4<f32>,42>;
};

[[group(1), binding(0)]]
var<uniform> global: Data;

fn function() -> vec4<f32> {
    var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var i: i32 = 0;

    loop {
        let _e9: i32 = i;
        if (!((_e9 < 42))) {
            break;
        }
        {
            let _e15: vec4<f32> = sum;
            let _e16: i32 = i;
            let _e18: vec4<f32> = global.vecs[_e16];
            sum = (_e15 + _e18);
        }
        continuing {
            let _e12: i32 = i;
            i = (_e12 + 1);
        }
    }
    let _e20: vec4<f32> = sum;
    return _e20;
}

fn main1() {
    return;
}

[[stage(vertex)]]
fn main() {
    main1();
    return;
}

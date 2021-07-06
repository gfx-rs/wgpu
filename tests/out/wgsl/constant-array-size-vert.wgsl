[[block]]
struct Data {
    vecs: [[stride(16)]] array<vec4<f32>,42>;
};

[[group(1), binding(0)]]
var<uniform> global: Data;

fn function() -> vec4<f32> {
    var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var i: i32 = 0;
    var local: i32;

    loop {
        let _e9: i32 = i;
        if (!((_e9 < 42))) {
            break;
        }
        {
            let _e17: vec4<f32> = sum;
            let _e18: i32 = i;
            let _e20: vec4<f32> = global.vecs[_e18];
            sum = (_e17 + _e20);
        }
        continuing {
            let _e12: i32 = i;
            local = _e12;
            i = (_e12 + 1);
        }
    }
    let _e22: vec4<f32> = sum;
    return _e22;
}


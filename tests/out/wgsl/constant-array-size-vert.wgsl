[[block]]
struct Data {
    vecs: [[stride(16)]] array<vec4<f32>,42u>;
};

[[group(1), binding(0)]]
var<uniform> global: Data;

fn function_() -> vec4<f32> {
    var sum: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var i: i32 = 0;

    loop {
        let e9: i32 = i;
        if (!((e9 < 42))) {
            break;
        }
        {
            let e15: vec4<f32> = sum;
            let e16: i32 = i;
            let e18: vec4<f32> = global.vecs[e16];
            sum = (e15 + e18);
        }
        continuing {
            let e12: i32 = i;
            i = (e12 + 1);
        }
    }
    let e20: vec4<f32> = sum;
    return e20;
}

fn main_1() {
    return;
}

[[stage(vertex)]]
fn main() {
    main_1();
    return;
}

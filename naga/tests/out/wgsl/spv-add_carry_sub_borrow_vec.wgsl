struct Output {
    sum: vec2<u32>,
    carry: vec2<u32>,
    diff: vec2<u32>,
    borrow: vec2<u32>,
}

struct Input {
    a: vec2<u32>,
    b: vec2<u32>,
}

@group(0) @binding(1) 
var<storage, read_write> outp: Output;
@group(0) @binding(0) 
var<storage, read_write> inp: Input;

fn main_1() {
    var c: vec2<u32>;
    var d: vec2<u32>;

    let _e5 = inp.a;
    let _e7 = inp.b;
    let _e8 = (_e5 + _e7);
    let _e11 = Input(_e8, vec2<u32>((_e8 < _e5)));
    c = _e11.b;
    outp.sum = _e11.a;
    let _e15 = c;
    outp.carry = _e15;
    let _e18 = inp.a;
    let _e20 = inp.b;
    let _e24 = Input((_e18 - _e20), vec2<u32>((_e18 < _e20)));
    d = _e24.b;
    outp.diff = _e24.a;
    let _e28 = d;
    outp.borrow = _e28;
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    main_1();
}

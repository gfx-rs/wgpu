@group(0) @binding(0) var<storage, read_write> out_u64_: u64;
@group(0) @binding(1) var<storage, read_write> out_i64_: i64;
@group(0) @binding(2) var<storage, read_write> out_vec_u64_: vec2<u64>;
@group(0) @binding(3) var<storage, read_write> out_vec_i64_: vec2<i64>;

@compute @workgroup_size(1)
fn main() {
    let a: u64 = 10lu;
    let b: u64 = 3lu;
    let c: i64 = -10li;
    let d: i64 = 3li;
    out_u64_ = (a / b) + (a % b);
    out_i64_ = (c / d) + (c % d);

    let va = vec2<u64>(a);
    let vb = vec2<u64>(b);
    let vc = vec2<i64>(c);
    let vd = vec2<i64>(d);
    out_vec_u64_ = (va / vb) + (va % vb);
    out_vec_i64_ = (vc / vd) + (vc % vd);
}

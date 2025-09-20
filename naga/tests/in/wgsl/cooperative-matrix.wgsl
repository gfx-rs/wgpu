var<private> a: coop_mat8x8<f32, A>;
var<private> b: coop_mat8x8<f32, B>;

@compute @workgroup_size(8, 8, 1)
fn main() {
    //let c = a * b;
}

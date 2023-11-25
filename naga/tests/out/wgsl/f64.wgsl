const k: f64 = 2.0lf;

var<private> v: f64 = 1.0lf;

fn f(x: f64) -> f64 {
    var z: f64;

    let y = (30.0lf + 400.0lf);
    z = (y + 5.0lf);
    return (((x + y) + k) + 5.0lf);
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    let _e1 = f(6.0lf);
    return;
}

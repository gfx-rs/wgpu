var<private> v: f64 = 1lf;
const k: f64 = 2.0lf;

fn f(x: f64) -> f64 {
   let y: f64 = 3e1lf + 4.0e2lf;
   var z = y + f64(5);
   return x + y + k + 5.0lf;
}

@compute @workgroup_size(1)
fn main() {
   f(6.0lf);
}

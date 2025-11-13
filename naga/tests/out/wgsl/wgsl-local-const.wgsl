const gb: i32 = 4i;
const gc: u32 = 4u;
const gd: f32 = 4f;

fn const_in_fn() {
    return;
}

@compute @workgroup_size(1, 1, 1) 
fn main() {
    const_in_fn();
    return;
}

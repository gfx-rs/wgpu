struct Inputs {
    @location(0) @interpolate(flat) a: vec3<i32>,
    @location(1) @interpolate(flat) b: vec3<i32>,
    @location(2) @interpolate(flat) choice: i32,
}

@fragment
fn main(in: Inputs) -> @location(0) vec3<i32> {
    return select(in.a, in.b, in.choice != 0);
}

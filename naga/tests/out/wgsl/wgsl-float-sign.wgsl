@group(0) @binding(0)
var<storage, read_write> floats: array<f32, 4>;
@group(0) @binding(1)
var<storage, read_write> ints: array<i32, 4>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    let _e4 = floats[1];
    floats[0] = sign(_e4);
    let _e8 = floats[2];
    let _e11 = floats[3];
    let v = vec2<f32>(_e8, _e11);
    let signs = sign(v);
    floats[2] = signs.x;
    floats[3] = signs.y;
    let _e24 = ints[1];
    ints[0] = sign(_e24);
    return;
}

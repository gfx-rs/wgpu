@group(0) @binding(0)
var<storage, read_write> out: array<i32, 2>;

@compute @workgroup_size(1)
fn main() {
    var tmp: array<i32, 1 << 1>=array(1, 2);
    for (var i = 0; i < 2; i++) {
        out[i] = tmp[i];
    }
}

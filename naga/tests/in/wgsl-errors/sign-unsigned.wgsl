@compute @workgroup_size(1)
fn main() {
    var input = vec4<u32>(1u);
    let value = sign(input);
}

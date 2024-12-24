@compute @workgroup_size(1, 1, 1) 
fn main() {
    let phony = unpack4xI8(12u)[2i];
    let phony_1 = unpack4xU8(12u).y;
}

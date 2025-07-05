fn test_packed_integer_dot_product() -> u32 {
    let a_5 = 1u;
    let b_5 = 2u;
    let c_5: i32 = dot4I8Packed(a_5, b_5);

    let a_6 = 3u;
    let b_6 = 4u;
    let c_6: u32 = dot4U8Packed(a_6, b_6);

    // test baking of arguments
    let c_7: i32 = dot4I8Packed(5u + c_6, 6u + c_6);
    let c_8: u32 = dot4U8Packed(7u + c_6, 8u + c_6);
    return c_8;
}

@compute @workgroup_size(1)
fn main() {
    let c = test_packed_integer_dot_product();
}

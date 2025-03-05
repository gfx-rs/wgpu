use wgpu::{DownlevelFlags, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[allow(non_snake_case)]
fn create_unpack4xU8_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let input: u32 = 0xAABBCCDD;
    let output: [u32; 4] = [0xDD, 0xCC, 0xBB, 0xAA];
    let unpack_u8 = ShaderTest::new(
        format!("unpack4xU8({input:X}) == {output:X?}"),
        String::from("value: u32"),
        String::from(
            "
                let a = unpack4xU8(input.value);
                output[0] = a[0];
                output[1] = a[1];
                output[2] = a[2];
                output[3] = a[3];
            ",
        ),
        &[input],
        &output,
    );
    tests.push(unpack_u8);

    tests
}

#[gpu_test]
static UNPACK4xU8: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_unpack4xU8_test())
    });

#[allow(non_snake_case)]
fn create_unpack4xI8_test() -> Vec<ShaderTest> {
    let mut tests = Vec::with_capacity(2);

    let values = [
        // regular unpacking
        (0x11223344, [0x44, 0x33, 0x22, 0x11]),
        // sign extension
        (0xFF, [-1, 0, 0, 0]),
    ];

    for (input, output) in values {
        let unpack_i8 = ShaderTest::new(
            format!("unpack4xI8({input:X}) == {output:X?}"),
            String::from("value: u32"),
            String::from(
                "
                    let a = bitcast<vec4<u32>>(unpack4xI8(input.value));
                    output[0] = a[0];
                    output[1] = a[1];
                    output[2] = a[2];
                    output[3] = a[3];
                ",
            ),
            &[input],
            &output,
        );
        tests.push(unpack_i8);
    }

    tests
}

#[gpu_test]
static UNPACK4xI8: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_unpack4xI8_test())
    });

#[allow(non_snake_case)]
fn create_pack4xU8_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    let input: [u32; 4] = [0xDD, 0xCC, 0xBB, 0xAA];
    let output: u32 = 0xAABBCCDD;
    let pack_u8 = ShaderTest::new(
        format!("pack4xU8({input:X?}) == {output:X}"),
        String::from("value: vec4<u32>"),
        String::from("output[0] = pack4xU8(input.value);"),
        &input,
        &[output],
    );
    tests.push(pack_u8);

    tests
}

#[gpu_test]
static PACK4xU8: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_pack4xU8_test())
    });

#[allow(non_snake_case)]
fn create_pack4xI8_test() -> Vec<ShaderTest> {
    let mut tests = Vec::with_capacity(2);

    let values: [([i32; 4], u32); 2] = [
        ([0x44, 0x33, 0x22, 0x11], 0x11223344),
        // Since the bit representation of the last 8 bits of each number in the input is the same
        // as the previous test's input numbers, the output should be equal
        ([-0xBB - 1, -0xCC - 1, -0xDD - 1, -0xEE - 1], 0x11223344),
    ];
    // Assure that test data of the first two cases end in equal bit values
    for value in values.map(|value| value.0)[..2].chunks_exact(2) {
        let [first, second] = value else {
            panic!("Expected at least 2 test values")
        };
        for (first, second) in first.iter().zip(second.iter()) {
            assert_eq!(
                first & 0xFF,
                second & 0xFF,
                "Last 8 bits of test values must be equal"
            );
        }
    }
    for (input, output) in values {
        let pack_i8 = ShaderTest::new(
            format!("pack4xI8({input:X?}) == {output:X}"),
            String::from("value: vec4<i32>"),
            String::from("output[0] = pack4xI8(input.value);"),
            &input,
            &[output],
        );
        tests.push(pack_i8);
    }

    tests
}

#[gpu_test]
static PACK4xI8: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_async(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_pack4xI8_test())
    });

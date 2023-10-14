use wgpu::{DownlevelFlags, Limits};

use crate::shader::{shader_input_output_test, InputStorageType, ShaderTest};
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

fn create_clamp_builtin_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    #[rustfmt::skip]
    let clamp_values: &[(f32, f32, f32, &[f32])] = &[
        // value - low - high - valid outputs

        // normal clamps
        (   20.0,  0.0,  10.0,  &[10.0]),
        (  -10.0,  0.0,  10.0,  &[0.0]),
        (    5.0,  0.0,  10.0,  &[5.0]),

        // med-of-three or min/max
        (    3.0,  2.0,  1.0,   &[1.0, 2.0]),
    ];

    for &(input, low, high, output) in clamp_values {
        let mut test = ShaderTest::new(
            format!("clamp({input}, 0.0, 10.0) == {output:?})"),
            String::from("value: f32, low: f32, high: f32"),
            String::from("output[0] = bitcast<u32>(clamp(input.value, input.low, input.high));"),
            &[input, low, high],
            &[output[0]],
        );
        for &extra in &output[1..] {
            test = test.extra_output_values(&[extra]);
        }

        tests.push(test);
    }

    tests
}

#[gpu_test]
static CLAMP_BUILTIN: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_clamp_builtin_test());
    });

fn create_pack_builtin_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    #[derive(Clone, Copy)]
    enum Function {
        Pack2x16Float,
        Pack2x16Unorm,
        Pack2x16Snorm,
        Pack4x8Snorm,
        Pack4x8Unorm,
    }

    #[rustfmt::skip]
    let values: &[(Function, &[f32], &[u32])] = &[
        (Function::Pack2x16Float, &[0., 0.], &[0x00000000]),
        (Function::Pack2x16Float, &[1., 0.], &[0x00003c00]),
        (Function::Pack2x16Float, &[1., 1.], &[0x3c003c00]),
        (Function::Pack2x16Float, &[-1., -1.], &[0xbc00bc00]),
        (Function::Pack2x16Float, &[10., 1.], &[0x3c004900]),
        (Function::Pack2x16Float, &[-10., 1.], &[0x3c00c900]),
        // f32 normal, but not f16 precise
        (Function::Pack2x16Float, &[1.00000011920928955078125, 1.], &[0x3c003c00]),
        // f32 subnormals
        (Function::Pack2x16Float, &[f32::MIN_POSITIVE * 0.1, 1.], &[0x3c000000, 0x3c008000, 0x3c000001]),
        (Function::Pack2x16Float, &[f32::MIN_POSITIVE * -0.1, 1.], &[0x3c008001, 0x3c000000, 0x3c008000]),

        // Unimplemented tests
        // (Function::Pack2x16Float, (f16::MIN_POSITIVE * 0.1) as f32, 1., &[0x00000000]),   // [0x3c0003ff, 0x3c000000, 0x3c008000]
        // (Function::Pack2x16Float, (f16::MIN_POSITIVE * -0.1) as f32, 1., &[0x00000000]),  // [0x03c0083ff, 0x3c000000, 0x3c008000]
        // (Function::Pack2x16Float, f32::MAX, 1., &[0x00000000]),   // TODO: Undefined
        // (Function::Pack2x16Float, f32::MIN, 1., &[0x00000000]),   // TODO: Undefined
        // (Function::Pack2x16Float, 1., f32::MAX, &[0x00000000]),   // TODO: Undefined
        // (Function::Pack2x16Float, 1., f32::MIN, &[0x00000000]),   // TODO: Undefined

        (Function::Pack2x16Snorm, &[0., 0.], &[0x00000000]),
        (Function::Pack2x16Snorm, &[1., 0.], &[0x00007fff]),
        (Function::Pack2x16Snorm, &[0., 1.], &[0x7fff0000]),
        (Function::Pack2x16Snorm, &[1., 1.], &[0x7fff7fff]),
        (Function::Pack2x16Snorm, &[-1., -1.], &[0x80018001]),
        (Function::Pack2x16Snorm, &[10., 10.], &[0x7fff7fff]),
        (Function::Pack2x16Snorm, &[-10., -10.], &[0x80018001]),
        (Function::Pack2x16Snorm, &[0.1, 0.1], &[0x0ccd0ccd]),
        (Function::Pack2x16Snorm, &[-0.1, -0.1], &[0xf333f333]),
        (Function::Pack2x16Snorm, &[0.5, 0.5], &[0x40004000]),
        (Function::Pack2x16Snorm, &[-0.5, -0.5], &[0xc000c000, 0xc001c001]), // rounding behavior of -0.5 is unspecified
        (Function::Pack2x16Snorm, &[0.1, 0.5], &[0x40000ccd]),
        (Function::Pack2x16Snorm, &[-0.1, -0.5], &[0xc000f333, 0xc001f333]), // rounding behavior of -0.5 is unspecified
        // subnormals
        (Function::Pack2x16Snorm, &[f32::MIN_POSITIVE * 0.1, 1.], &[0x7fff0000]),
        (Function::Pack2x16Snorm, &[f32::MIN_POSITIVE * -0.1, 1.], &[0x7fff0000]),

        (Function::Pack2x16Unorm, &[0., 0.], &[0x00000000]),
        (Function::Pack2x16Unorm, &[1., 0.], &[0x0000ffff]),
        (Function::Pack2x16Unorm, &[0., 1.], &[0xffff0000]),
        (Function::Pack2x16Unorm, &[1., 1.], &[0xffffffff]),
        (Function::Pack2x16Unorm, &[-1., -1.], &[0x00000000]),
        (Function::Pack2x16Unorm, &[0.1, 0.1], &[0x199a199a]),
        (Function::Pack2x16Unorm, &[0.5, 0.5], &[0x80008000]),
        (Function::Pack2x16Unorm, &[0.1, 0.5], &[0x8000199a]),
        (Function::Pack2x16Unorm, &[10., 10.], &[0xffffffff]),
        // subnormals
        (Function::Pack2x16Unorm, &[f32::MIN_POSITIVE * 0.1, 1.], &[0xffff0000]),

        // Normals
        (Function::Pack4x8Snorm, &[0., 0., 0., 0.], &[0x00000000]),
        (Function::Pack4x8Snorm, &[1., 0., 0., 0.], &[0x0000007f]),
        (Function::Pack4x8Snorm, &[0., 1., 0., 0.], &[0x00007f00]),
        (Function::Pack4x8Snorm, &[0., 0., 1., 0.], &[0x007f0000]),
        (Function::Pack4x8Snorm, &[0., 0., 0., 1.], &[0x7f000000]),
        (Function::Pack4x8Snorm, &[1., 1., 1., 1.], &[0x7f7f7f7f]),
        (Function::Pack4x8Snorm, &[10., 10., 10., 10.], &[0x7f7f7f7f]),
        (Function::Pack4x8Snorm, &[-1., 0., 0., 0.], &[0x00000081]),
        (Function::Pack4x8Snorm, &[0., -1., 0., 0.], &[0x00008100]),
        (Function::Pack4x8Snorm, &[0., 0., -1., 0.], &[0x00810000]),
        (Function::Pack4x8Snorm, &[0., 0., 0., -1.], &[0x81000000]),
        (Function::Pack4x8Snorm, &[-1., -1., -1., -1.], &[0x81818181]),
        (Function::Pack4x8Snorm, &[-10., -10., -10., -10.], &[0x81818181]),
        (Function::Pack4x8Snorm, &[0.1, 0.1, 0.1, 0.1], &[0x0d0d0d0d]),
        (Function::Pack4x8Snorm, &[-0.1, -0.1, -0.1, -0.1], &[0xf3f3f3f3]),
        (Function::Pack4x8Snorm, &[0.1, -0.1, 0.1, -0.1], &[0xf30df30d]),
        (Function::Pack4x8Snorm, &[0.5, 0.5, 0.5, 0.5], &[0x40404040]),
        (Function::Pack4x8Snorm, &[-0.5, -0.5, -0.5, -0.5], &[0xc0c0c0c0, 0xc1c1c1c1]), // rounding behavior of -0.5 is unspecified
        (Function::Pack4x8Snorm, &[-0.5, 0.5, -0.5, 0.5], &[0x40c040c0, 0x40c140c1]), // rounding behavior of -0.5 is unspecified
        (Function::Pack4x8Snorm, &[0.1, 0.5, 0.1, 0.5], &[0x400d400d]),
        (Function::Pack4x8Snorm, &[-0.1, -0.5, -0.1, -0.5], &[0xc0f3c0f3, 0xc1f3c1f3]), // rounding behavior of -0.5 is unspecified
        // Subnormals
        (Function::Pack4x8Snorm, &[f32::MIN_POSITIVE * 0.1, 1., 1., 1.], &[0x7f7f7f00]),
        (Function::Pack4x8Snorm, &[f32::MIN_POSITIVE * -0.1, 1., 1., 1.], &[0x7f7f7f00]),
        
        (Function::Pack4x8Unorm, &[0., 0., 0., 0.], &[0x00000000]),
        (Function::Pack4x8Unorm, &[1., 0., 0., 0.], &[0x000000ff]),
        (Function::Pack4x8Unorm, &[0., 1., 0., 0.], &[0x0000ff00]),
        (Function::Pack4x8Unorm, &[0., 0., 1., 0.], &[0x00ff0000]),
        (Function::Pack4x8Unorm, &[0., 0., 0., 1.], &[0xff000000]),
        (Function::Pack4x8Unorm, &[1., 1., 1., 1.], &[0xffffffff]),
        (Function::Pack4x8Unorm, &[10., 10., 10., 10.], &[0xffffffff]),
        (Function::Pack4x8Unorm, &[-1., -1., -1., -1.], &[0x00000000]),
        (Function::Pack4x8Unorm, &[-10., -10., -10., -10.], &[0x00000000]),
        (Function::Pack4x8Unorm, &[0.1, 0.1, 0.1, 0.1], &[0x1a1a1a1a]),
        (Function::Pack4x8Unorm, &[0.5, 0.5, 0.5, 0.5], &[0x80808080]),
        (Function::Pack4x8Unorm, &[0.1, 0.5, 0.1, 0.5], &[0x801a801a]),
        // subnormals
        (Function::Pack4x8Unorm, &[f32::MIN_POSITIVE * 0.1, 1., 1., 1.], &[0xffffff00]),
    ];

    for &(function, inputs, outputs) in values {
        let name = match function {
            Function::Pack2x16Float => format!("pack2x16float({inputs:?}) == {outputs:#x?}"),
            Function::Pack2x16Unorm => format!("pack2x16unorm({inputs:?}) == {outputs:#x?}"),
            Function::Pack2x16Snorm => format!("pack2x16snorm({inputs:?}) == {outputs:#x?}"),
            Function::Pack4x8Snorm => format!("pack4x8snorm({inputs:?}) == {outputs:#x?}"),
            Function::Pack4x8Unorm => format!("pack4x8unorm({inputs:?}) == {outputs:#x?}"),
        };

        let members = match function {
            Function::Pack2x16Float => "value: vec2f",
            Function::Pack2x16Unorm => "value: vec2f",
            Function::Pack2x16Snorm => "value: vec2f",
            Function::Pack4x8Snorm => "value: vec4f",
            Function::Pack4x8Unorm => "value: vec4f",
        };

        let body = match function {
            Function::Pack2x16Float => "output[0] = pack2x16float(input.value);",
            Function::Pack2x16Unorm => "output[0] = pack2x16unorm(input.value);",
            Function::Pack2x16Snorm => "output[0] = pack2x16snorm(input.value);",
            Function::Pack4x8Snorm => "output[0] = pack4x8snorm(input.value);",
            Function::Pack4x8Unorm => "output[0] = pack4x8unorm(input.value);",
        };

        let mut test = ShaderTest::new(
            name,
            String::from(members),
            String::from(body),
            inputs,
            &[outputs[0]],
        );

        for &output in &outputs[1..] {
            test = test.extra_output_values(&[output]);
        }
        tests.push(test);
    }

    tests
}

#[gpu_test]
static PACKING_BUILTINS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_pack_builtin_test());
    });

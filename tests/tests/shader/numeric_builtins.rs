use std::ops::Range;

use wgpu::{DownlevelFlags, Limits};
use wgt::math::f32_next;

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

    for &(input, low, high, outputs) in clamp_values {
        let nested_outputs: Vec<_> = outputs.iter().map(|v| std::slice::from_ref(v)).collect();
        tests.push(ShaderTest::new(
            format!("clamp({input}, 0.0, 10.0) == {outputs:?})"),
            String::from("value: f32, low: f32, high: f32"),
            String::from("output[0] = bitcast<u32>(clamp(input.value, input.low, input.high));"),
            &[input, low, high],
            &nested_outputs,
        ));
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

#[allow(clippy::excessive_precision)]
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

        let outputs: Vec<_> = outputs.iter().map(|v| std::slice::from_ref(v)).collect();

        tests.push(ShaderTest::new(
            name,
            String::from(members),
            String::from(body),
            inputs,
            &outputs,
        ));
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

#[allow(clippy::excessive_precision)]
fn create_unpack_builtin_test() -> Vec<ShaderTest> {
    let mut tests = Vec::new();

    // Magic numbers from the spec
    // https://github.com/gpuweb/cts/blob/main/src/unittests/floating_point.spec.ts

    pub const ZERO_BOUNDS: Range<f32> = f32::MIN_POSITIVE * -1.0..f32::MIN_POSITIVE;

    pub const ONE_BOUNDS_SNORM: Range<f32> = 0.999999821186065673828125..1.0000002384185791015625;

    pub const ONE_BOUNDS_UNORM: Range<f32> =
        0.9999998509883880615234375..1.0000001490116119384765625;

    pub const NEG_ONE_BOUNDS_SNORM: Range<f32> = -1.0 - f32::EPSILON..-0.999999821186065673828125;

    pub const HALF_BOUNDS_2X16_SNORM: Range<f32> =
        0.500015079975128173828125..0.5000154972076416015625;

    pub const NEG_HALF_BOUNDS_2X16_SNORM: Range<f32> =
        -0.4999848306179046630859375..-0.49998462200164794921875;

    pub const HALF_BOUNDS_2X16_UNORM: Range<f32> =
        0.5000074803829193115234375..0.5000078380107879638671875;

    pub const HALF_BOUNDS_4X8_SNORM: Range<f32> =
        0.503936827182769775390625..0.503937244415283203125;

    pub const NEG_HALF_BOUNDS_4X8_SNORM: Range<f32> =
        -0.4960630834102630615234375..-0.49606287479400634765625;

    pub const HALF_BOUNDS_4X8_UNORM: Range<f32> =
        0.5019606053829193115234375..0.5019609630107879638671875;

    fn range(value: f32) -> Range<f32> {
        value..f32_next(value)
    }

    #[derive(Clone, Copy)]
    enum Function {
        Unpack2x16Float,
        Unpack2x16Unorm,
        Unpack2x16Snorm,
        Unpack4x8Snorm,
        Unpack4x8Unorm,
    }

    #[rustfmt::skip]
    let values: &[(Function, u32, &[Range<f32>])] = &[
        (Function::Unpack2x16Snorm, 0x00000000, &[ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack2x16Snorm, 0x00007fff, &[ONE_BOUNDS_SNORM, ZERO_BOUNDS]),
        (Function::Unpack2x16Snorm, 0x7fff0000, &[ZERO_BOUNDS, ONE_BOUNDS_SNORM]),
        (Function::Unpack2x16Snorm, 0x7fff7fff, &[ONE_BOUNDS_SNORM, ONE_BOUNDS_SNORM]),
        (Function::Unpack2x16Snorm, 0x80018001, &[NEG_ONE_BOUNDS_SNORM, NEG_ONE_BOUNDS_SNORM]),
        (Function::Unpack2x16Snorm, 0x40004000, &[HALF_BOUNDS_2X16_SNORM, HALF_BOUNDS_2X16_SNORM]),
        (Function::Unpack2x16Snorm, 0xc001c001, &[NEG_HALF_BOUNDS_2X16_SNORM, NEG_HALF_BOUNDS_2X16_SNORM]),
        (Function::Unpack2x16Snorm, 0x0000c001, &[NEG_HALF_BOUNDS_2X16_SNORM, ZERO_BOUNDS]),
        (Function::Unpack2x16Snorm, 0xc0010000, &[ZERO_BOUNDS, NEG_HALF_BOUNDS_2X16_SNORM]),

        (Function::Unpack2x16Unorm, 0x00000000, &[ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack2x16Unorm, 0x0000ffff, &[ONE_BOUNDS_UNORM, ZERO_BOUNDS]),
        (Function::Unpack2x16Unorm, 0xffff0000, &[ZERO_BOUNDS, ONE_BOUNDS_UNORM]),
        (Function::Unpack2x16Unorm, 0xffffffff, &[ONE_BOUNDS_UNORM, ONE_BOUNDS_UNORM]),
        (Function::Unpack2x16Unorm, 0x80008000, &[HALF_BOUNDS_2X16_UNORM, HALF_BOUNDS_2X16_UNORM]),

        (Function::Unpack4x8Snorm, 0x00000000, &[ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Snorm, 0x0000007f, &[ONE_BOUNDS_SNORM, ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Snorm, 0x00007f00, &[ZERO_BOUNDS, ONE_BOUNDS_SNORM, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Snorm, 0x007f0000, &[ZERO_BOUNDS, ZERO_BOUNDS, ONE_BOUNDS_SNORM, ZERO_BOUNDS]),
        (Function::Unpack4x8Snorm, 0x7f000000, &[ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS, ONE_BOUNDS_SNORM]),
        (Function::Unpack4x8Snorm, 0x00007f7f, &[ONE_BOUNDS_SNORM, ONE_BOUNDS_SNORM, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Snorm, 0x7f7f0000, &[ZERO_BOUNDS, ZERO_BOUNDS, ONE_BOUNDS_SNORM, ONE_BOUNDS_SNORM]),
        (Function::Unpack4x8Snorm, 0x7f007f00, &[ZERO_BOUNDS, ONE_BOUNDS_SNORM, ZERO_BOUNDS, ONE_BOUNDS_SNORM]),
        (Function::Unpack4x8Snorm, 0x007f007f, &[ONE_BOUNDS_SNORM, ZERO_BOUNDS, ONE_BOUNDS_SNORM, ZERO_BOUNDS]),
        (Function::Unpack4x8Snorm, 0x7f7f7f7f, &[ONE_BOUNDS_SNORM, ONE_BOUNDS_SNORM, ONE_BOUNDS_SNORM, ONE_BOUNDS_SNORM]),
        (Function::Unpack4x8Snorm, 0x81818181, &[NEG_ONE_BOUNDS_SNORM, NEG_ONE_BOUNDS_SNORM, NEG_ONE_BOUNDS_SNORM, NEG_ONE_BOUNDS_SNORM]),
        (Function::Unpack4x8Snorm, 0x40404040, &[HALF_BOUNDS_4X8_SNORM, HALF_BOUNDS_4X8_SNORM, HALF_BOUNDS_4X8_SNORM, HALF_BOUNDS_4X8_SNORM]),
        (Function::Unpack4x8Snorm, 0xc1c1c1c1, &[NEG_HALF_BOUNDS_4X8_SNORM, NEG_HALF_BOUNDS_4X8_SNORM, NEG_HALF_BOUNDS_4X8_SNORM, NEG_HALF_BOUNDS_4X8_SNORM]),

        (Function::Unpack4x8Unorm, 0x00000000, &[ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Unorm, 0x000000ff, &[ONE_BOUNDS_UNORM, ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Unorm, 0x0000ff00, &[ZERO_BOUNDS, ONE_BOUNDS_UNORM, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Unorm, 0x00ff0000, &[ZERO_BOUNDS, ZERO_BOUNDS, ONE_BOUNDS_UNORM, ZERO_BOUNDS]),
        (Function::Unpack4x8Unorm, 0xff000000, &[ZERO_BOUNDS, ZERO_BOUNDS, ZERO_BOUNDS, ONE_BOUNDS_UNORM]),
        (Function::Unpack4x8Unorm, 0x0000ffff, &[ONE_BOUNDS_UNORM, ONE_BOUNDS_UNORM, ZERO_BOUNDS, ZERO_BOUNDS]),
        (Function::Unpack4x8Unorm, 0xffff0000, &[ZERO_BOUNDS, ZERO_BOUNDS, ONE_BOUNDS_UNORM, ONE_BOUNDS_UNORM]),
        (Function::Unpack4x8Unorm, 0xff00ff00, &[ZERO_BOUNDS, ONE_BOUNDS_UNORM, ZERO_BOUNDS, ONE_BOUNDS_UNORM]),
        (Function::Unpack4x8Unorm, 0x00ff00ff, &[ONE_BOUNDS_UNORM, ZERO_BOUNDS, ONE_BOUNDS_UNORM, ZERO_BOUNDS]),
        (Function::Unpack4x8Unorm, 0xffffffff, &[ONE_BOUNDS_UNORM, ONE_BOUNDS_UNORM, ONE_BOUNDS_UNORM, ONE_BOUNDS_UNORM]),
        (Function::Unpack4x8Unorm, 0x80808080, &[HALF_BOUNDS_4X8_UNORM, HALF_BOUNDS_4X8_UNORM, HALF_BOUNDS_4X8_UNORM, HALF_BOUNDS_4X8_UNORM]),


        (Function::Unpack2x16Float, 0x00000000, &[range(0.0), range(0.0)]),
        (Function::Unpack2x16Float, 0x80000000, &[range(0.0), range(0.0)]),
        (Function::Unpack2x16Float, 0x00008000, &[range(0.0), range(0.0)]),
        (Function::Unpack2x16Float, 0x80008000, &[range(0.0), range(0.0)]),
        (Function::Unpack2x16Float, 0x00003c00, &[range(1.0), range(0.0)]),
        (Function::Unpack2x16Float, 0x3c000000, &[range(0.0), range(1.0)]),
        (Function::Unpack2x16Float, 0x3c003c00, &[range(1.0), range(1.0)]),
        (Function::Unpack2x16Float, 0xbc00bc00, &[range(-1.0), range(-1.0)]),
        (Function::Unpack2x16Float, 0x49004900, &[range(10.0), range(10.0)]),
        (Function::Unpack2x16Float, 0xc900c900, &[range(-10.0), range(-10.0)]),
    ];

    for &(function, input, outputs) in values {
        let name = match function {
            Function::Unpack2x16Float => format!("unpack2x16float({input:#x?}) == {outputs:?}"),
            Function::Unpack2x16Unorm => format!("unpack2x16unorm({input:#x?}) == {outputs:?}"),
            Function::Unpack2x16Snorm => format!("unpack2x16snorm({input:#x?}) == {outputs:?}"),
            Function::Unpack4x8Snorm => format!("unpack4x8snorm({input:#x?}) == {outputs:?}"),
            Function::Unpack4x8Unorm => format!("unpack4x8unorm({input:#x?}) == {outputs:?}"),
        };

        let body = match function {
            Function::Unpack2x16Float => {
                "
                let value = unpack2x16float(input.value);
                output[0] = bitcast<u32>(value.x);
                output[1] = bitcast<u32>(value.y);
            "
            }
            Function::Unpack2x16Unorm => {
                "
                let value = unpack2x16unorm(input.value);
                output[0] = bitcast<u32>(value.x);
                output[1] = bitcast<u32>(value.y);
            "
            }
            Function::Unpack2x16Snorm => {
                "
                let value = unpack2x16snorm(input.value);
                output[0] = bitcast<u32>(value.x);
                output[1] = bitcast<u32>(value.y);
            "
            }
            Function::Unpack4x8Snorm => {
                "
                let value = unpack4x8snorm(input.value);
                output[0] = bitcast<u32>(value.x);
                output[1] = bitcast<u32>(value.y);
                output[2] = bitcast<u32>(value.z);
                output[3] = bitcast<u32>(value.w);
            "
            }
            Function::Unpack4x8Unorm => {
                "
                let value = unpack4x8unorm(input.value);
                output[0] = bitcast<u32>(value.x);
                output[1] = bitcast<u32>(value.y);
                output[2] = bitcast<u32>(value.z);
                output[3] = bitcast<u32>(value.w);
            "
            }
        };

        tests.push(ShaderTest::new(
            name,
            String::from("value: u32"),
            String::from(body),
            &[input],
            &[outputs],
        ));
    }

    tests
}

#[gpu_test]
static UNPACKING_BUILTINS: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults()),
    )
    .run_sync(|ctx| {
        shader_input_output_test(ctx, InputStorageType::Storage, create_unpack_builtin_test());
    });

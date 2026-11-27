//! Direct f16 narrowing stores to storage buffers round to nearest even.

use half::f16;
use wgpu::{Backends, DownlevelFlags, Features, Limits};
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

use super::{shader_input_output_test, InputStorageType, ShaderTest};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(F16_STORE_ROUNDING);
}

fn create_f16_store_rounding_test() -> Vec<ShaderTest> {
    // Ties between representable f16 values around 1.0 (ulp = 2^-10):
    //   1 + 2^-11     -> 0x3C00 (even; RTZ also gives 0x3C00)
    //   1 + 3 * 2^-11 -> 0x3C02 (even; RTZ gives 0x3C01)
    //   just above the tie -> 0x3C01 (RTZ gives 0x3C00), just below -> 0x3C00
    //   -(1 + 3 * 2^-11) -> 0xBC02 (even; RTZ gives 0xBC01)
    // Upper finite exponent (ulp = 32.0, avoiding +/-Inf which is poison under
    // Vulkan's default `NotInf` without `SignedZeroInfNanPreserve`):
    //   65456.0 -> 0x7BFE = 65472.0 (even; RTZ gives 0x7BFD = 65440.0)
    //   65489.0 -> 0x7BFF = 65504.0 = f16::MAX (RTZ gives 0x7BFE = 65472.0)
    let tie_low = 1.0f32 + f32::powi(2.0, -11);
    let tie_high = 1.0f32 + 3.0 * f32::powi(2.0, -11);
    let above_tie = tie_low + f32::powi(2.0, -20);
    let below_tie = tie_low - f32::powi(2.0, -20);
    let cases = [
        ("positive ties", tie_low, tie_high),
        ("around tie", above_tie, below_tie),
        ("negative ties", -tie_low, -tie_high),
        ("near max finite", 65456.0, 65489.0),
    ];

    let mut tests = Vec::with_capacity(cases.len() * 2);
    for &(name, a, b) in &cases {
        let expected = [f16::from_f32(a).to_bits(), f16::from_f32(b).to_bits()];
        tests.push(
            ShaderTest::new(
                format!("{name} (vec2)"),
                "value: vec2<f32>".into(),
                "output = vec2<f16>(input.value);".into(),
                &[a, b],
                &expected,
            )
            .header("enable f16;".into())
            .output_type("vec2<f16>".into()),
        );
        tests.push(
            ShaderTest::new(
                format!("{name} (scalar)"),
                "value: vec2<f32>".into(),
                "output[0] = f16(input.value.x); output[1] = f16(input.value.y);".into(),
                &[a, b],
                &expected,
            )
            .header("enable f16;".into())
            .output_type("array<f16, 2>".into()),
        );
    }
    tests
}

#[apply(gpu_test!)]
static F16_STORE_ROUNDING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::SHADER_F16)
            .downlevel_flags(DownlevelFlags::COMPUTE_SHADERS)
            .limits(Limits::downlevel_defaults())
            // WGSL leaves f32 -> f16 rounding choice between adjacent representable
            // values unspecified (e.g. HLSL/DX12 may round toward zero). This test
            // exercises Naga's SPIR-V `FPRoundingMode RTE` decoration on Vulkan.
            .skip(FailureCase::backend(Backends::all() - Backends::VULKAN)),
    )
    .run_async(|ctx| {
        shader_input_output_test(
            ctx,
            InputStorageType::Storage,
            create_f16_store_rounding_test(),
        )
    });

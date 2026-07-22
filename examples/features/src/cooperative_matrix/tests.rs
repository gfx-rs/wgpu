use super::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

// These tests are split into `COOPERATIVE_MATRIX_F16` and
// `COOPERATIVE_MATRIX_F32` because the latter can run without
// `wgpu::Features::SHADER_F16`.

#[gpu_test]
pub static COOPERATIVE_MATRIX_F32: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            .limits(wgpu::Limits::default()),
    )
    .run_async(|ctx| async move {
        let coop_props = ctx.adapter.cooperative_matrix_properties();
        // `shader.wgsl` hardcodes 8x8 f32 tiles (`coop_mat8x8<f32, ...>`),
        // so only an exact match is usable here.
        let config = coop_props.iter().find(|prop| {
            prop.m_size == 8
                && prop.n_size == 8
                && prop.k_size == 8
                && prop.ab_type == wgpu::CooperativeScalarType::F32
                && prop.cr_type == wgpu::CooperativeScalarType::F32
        });
        let Some(config) = config else {
            // Not every adapter that supports EXPERIMENTAL_COOPERATIVE_MATRIX
            // exposes an 8x8x8 f32/f32 configuration -- e.g. tensor/matrix-core
            // hardware commonly multiplies in a reduced-precision input type
            // and only optionally accumulates at f32, so plain f32 inputs are
            // often unsupported. We can't `.skip()` this per-adapter without
            // that list growing without bound across every GPU architecture
            // contributors happen to test on, so we log and move on instead.
            log::warn!(
                "No 8x8x8 f32 cooperative matrix configuration found among: \
                 {coop_props:?}; skipping test"
            );
            return;
        };
        let ExecuteResults {
            max_error,
            tolerance,
            matrix: _,
        } = execute(&ctx.device, &ctx.queue, config).await;
        assert!(max_error < tolerance);
    });

#[gpu_test]
pub static COOPERATIVE_MATRIX_F16: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX | wgpu::Features::SHADER_F16)
            .limits(wgpu::Limits::default()),
    )
    .run_async(|ctx| async move {
        let coop_props = ctx.adapter.cooperative_matrix_properties();
        // `shader_f16_16x16.wgsl` hardcodes 16x16 f16 tiles
        // (`coop_mat16x16<f16, ...>`), so only an exact match is usable here.
        let config = coop_props.iter().find(|prop| {
            prop.m_size == 16
                && prop.n_size == 16
                && prop.k_size == 16
                && prop.ab_type == wgpu::CooperativeScalarType::F16
                && prop.cr_type == wgpu::CooperativeScalarType::F16
        });
        let Some(config) = config else {
            // See the comment in COOPERATIVE_MATRIX_F32 above: not every adapter
            // exposes a 16x16x16 f16/f16 configuration (e.g. some only pair
            // smaller tile sizes with f16), and we don't want a per-adapter
            // `.skip()` list to grow without bound, so we log and move on.
            log::warn!(
                "No 16x16x16 f16 cooperative matrix configuration found among: \
                 {coop_props:?}; skipping test"
            );
            return;
        };
        let ExecuteResults {
            max_error,
            tolerance,
            matrix: _,
        } = execute(&ctx.device, &ctx.queue, config).await;
        assert!(max_error < tolerance);
    });

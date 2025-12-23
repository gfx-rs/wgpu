use super::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters};

#[gpu_test]
pub static COOPERATIVE_MATRIX: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
            .limits(wgpu::Limits::default()),
    )
    .run_async(|ctx| async move {
        let coop_props = ctx.adapter.cooperative_matrix_properties();
        let config = coop_props
            .iter()
            .find(|prop| {
                prop.m_size == 16
                    && prop.n_size == 16
                    && prop.k_size == 16
                    && prop.ab_type == wgpu::CooperativeScalarType::F16
                    && prop.cr_type == wgpu::CooperativeScalarType::F16
            })
            .or_else(|| {
                coop_props.iter().find(|prop| {
                    prop.m_size == 8
                        && prop.n_size == 8
                        && prop.k_size == 8
                        && prop.ab_type == wgpu::CooperativeScalarType::F32
                        && prop.cr_type == wgpu::CooperativeScalarType::F32
                })
            })
            .unwrap();
        let ExecuteResults {
            max_error,
            tolerance,
            matrix: _,
        } = execute(&ctx.device, &ctx.queue, config).await;
        assert!(max_error < tolerance);
    });

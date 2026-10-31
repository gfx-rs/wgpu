use wgpu::Backends;
use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(STRICT_WEBGPU_COMPLIANCE_ADAPTER);
    vec.push(DX12_ADAPTER_LUID_FILTER);
}

#[apply(gpu_test!)]
static STRICT_WEBGPU_COMPLIANCE_ADAPTER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .instance_flags(wgpu::InstanceFlags::STRICT_WEBGPU_COMPLIANCE)
            .enable_noop(),
    )
    .run_sync(|ctx| {
        assert!(ctx
            .adapter
            .get_downlevel_capabilities()
            .is_webgpu_compliant());
        let mut limits = wgpu::Limits::defaults();
        limits.zero_native_only();
        assert!(limits.check_limits(&ctx.adapter.limits()));
    });

#[apply(gpu_test!)]
static DX12_ADAPTER_LUID_FILTER: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default().skip(FailureCase::backend(Backends::all() - Backends::DX12)),
    )
    .run_async(|ctx| async move {
        match ctx.adapter_info.backend {
            #[cfg(target_os = "windows")]
            wgpu::Backend::Dx12 => check_dx12_adapter_luid_filter(&ctx).await,
            other => unreachable!(
                "test is configured to skip all backends except Dx12, but ran on {other:?}"
            ),
        }
    });

#[cfg(target_os = "windows")]
async fn check_dx12_adapter_luid_filter(ctx: &wgpu_test::TestingContext) {
    use wgpu::hal::dx12;

    let adapter_luid = |adapter: &wgpu::Adapter| {
        let hal_adapter =
            unsafe { adapter.as_hal::<dx12::Api>() }.expect("adapter backend mismatch");
        unsafe { hal_adapter.as_raw().GetDesc() }
            .unwrap()
            .AdapterLuid
    };
    let options_for_luid = |luid| {
        wgpu::BackendAdapterOptionsMap::default().set(dx12::Dx12AdapterOptions { luid: Some(luid) })
    };

    let luid = adapter_luid(&ctx.adapter);
    let selected = unsafe {
        ctx.instance.request_adapter_ext(
            &wgpu::RequestAdapterOptions::default(),
            options_for_luid(luid),
        )
    }
    .await
    .expect("adapter should be selected by its LUID");
    assert_eq!(adapter_luid(&selected), luid);
    assert_eq!(selected.get_info(), ctx.adapter_info);

    let known_luids = ctx
        .instance
        .enumerate_adapters(Backends::DX12)
        .await
        .iter()
        .map(adapter_luid)
        .collect::<Vec<_>>();
    let mut unknown_luid = luid;
    while known_luids.contains(&unknown_luid) {
        unknown_luid.LowPart = unknown_luid.LowPart.wrapping_add(1);
    }
    let result = unsafe {
        ctx.instance.request_adapter_ext(
            &wgpu::RequestAdapterOptions::default(),
            options_for_luid(unknown_luid),
        )
    }
    .await;
    assert!(
        matches!(result, Err(wgpu::RequestAdapterError::NotFound { .. })),
        "expected no adapter for unknown LUID, got {result:?}"
    );
}

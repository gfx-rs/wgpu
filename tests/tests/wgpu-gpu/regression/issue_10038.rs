use wgpu_test::{
    apply, gpu_test, FailureCase, GpuTestConfiguration, GpuTestInitializer, TestParameters,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(TEXTURE_COUNTERS_BALANCED);
}

#[apply(gpu_test!)]
static TEXTURE_COUNTERS_BALANCED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            // NOTE: The WebGPU backend does not implement internal counters.
            .skip(FailureCase::backend(wgpu::Backends::BROWSER_WEBGPU)),
    )
    .run_async(|ctx| async move {
        let before = ctx.device.get_internal_counters().hal;

        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("internal counters test"),
            size: wgpu::Extent3d {
                width: 256,
                height: 256,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let alive = ctx.device.get_internal_counters().hal;
        assert_eq!(
            alive.textures.read(),
            before.textures.read() + 1,
            "internal texture counter should increment with new texture",
        );

        drop(texture);
        ctx.async_poll(wgpu::PollType::wait_indefinitely())
            .await
            .unwrap();

        let after = ctx.device.get_internal_counters().hal;
        assert_eq!(
            after.textures.read(),
            before.textures.read(),
            "internal texture counter should return to baseline once texture is destroyed",
        );
        assert_eq!(
            after.texture_memory.read(),
            before.texture_memory.read(),
            concat!(
                "internal texture memory counter should return to its baseline ",
                "once the texture is destroyed"
            ),
        );
    });

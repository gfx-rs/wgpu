use wgpu_test::{fail, gpu_test, valid, GpuTestConfiguration};

#[gpu_test]
static BAD_BUFFER: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    // Create a buffer with bad parameters and call a few methods.
    // Validation should fail but there should be not panic.
    let buffer = fail(
        &ctx.device,
        || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 99999999,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        },
        Some("`map` usage can only be combined with the opposite `copy`"),
    );

    let error = match ctx.adapter_info.backend.to_str() {
        "vulkan" | "vk" => "bufferid id(0,1,vk) is invalid",
        "dx12" | "d3d12" => "bufferid id(0,1,d3d12) is invalid",
        "metal" | "mtl" => "bufferid id(0,1,mtl) is invalid",
        "opengl" | "gles" | "gl" => "bufferid id(0,1,gl) is invalid",
        "webgpu" => "bufferid id(0,1,webgpu) is invalid",
        b => b,
    };

    fail(
        &ctx.device,
        || buffer.slice(..).map_async(wgpu::MapMode::Write, |_| {}),
        Some(error),
    );
    fail(&ctx.device, || buffer.unmap(), Some(error));
    valid(&ctx.device, || buffer.destroy());
    valid(&ctx.device, || buffer.destroy());
});

#[gpu_test]
static BAD_TEXTURE: GpuTestConfiguration = GpuTestConfiguration::new().run_sync(|ctx| {
    let texture = fail(
        &ctx.device,
        || {
            ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d {
                    width: 0,
                    height: 12345678,
                    depth_or_array_layers: 9001,
                },
                mip_level_count: 2000,
                sample_count: 27,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::all(),
                view_formats: &[],
            })
        },
        Some("dimension x is zero"),
    );

    let error = match ctx.adapter_info.backend.to_str() {
        "vulkan" | "vk" => "textureid id(0,1,vk) is invalid",
        "dx12" | "d3d12" => "textureid id(0,1,d3d12) is invalid",
        "metal" | "mtl" => "textureid id(0,1,mtl) is invalid",
        "opengl" | "gles" | "gl" => "textureid id(0,1,gl) is invalid",
        "webgpu" => "textureid id(0,1,webgpu) is invalid",
        b => b,
    };

    fail(
        &ctx.device,
        || {
            let _ = texture.create_view(&wgpu::TextureViewDescriptor::default());
        },
        Some(error),
    );
    valid(&ctx.device, || texture.destroy());
    valid(&ctx.device, || texture.destroy());
});

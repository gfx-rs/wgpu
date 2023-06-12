use wgpu_test::{fail, infra::GpuTest, valid};

#[derive(Default)]
pub struct BadBufferTest;

impl GpuTest for BadBufferTest {
    fn run(&self, ctx: wgpu_test::TestingContext) {
        // Create a buffer with bad parameters and call a few methods.
        // Validation should fail but there should be not panic.
        let buffer = fail(&ctx.device, || {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 99999999,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });

        fail(&ctx.device, || {
            buffer.slice(..).map_async(wgpu::MapMode::Write, |_| {})
        });
        fail(&ctx.device, || buffer.unmap());
        valid(&ctx.device, || buffer.destroy());
        valid(&ctx.device, || buffer.destroy());
    }
}

#[derive(Default)]
pub struct BadTextureTest;

impl GpuTest for BadTextureTest {
    fn run(&self, ctx: wgpu_test::TestingContext) {
        let texture = fail(&ctx.device, || {
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
        });

        fail(&ctx.device, || {
            let _ = texture.create_view(&wgpu::TextureViewDescriptor::default());
        });
        valid(&ctx.device, || texture.destroy());
        valid(&ctx.device, || texture.destroy());
    }
}

use wasm_bindgen_test::*;
use wgpu_test::{fail, initialize_test, valid, TestParameters};

#[test]
#[wasm_bindgen_test]
fn bad_buffer() {
    // Create a buffer with bad parameters and call a few methods.
    // Validation should fail but there should be not panic.
    initialize_test(TestParameters::default(), |ctx| {
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
    });
}

#[test]
#[wasm_bindgen_test]
fn bad_texture() {
    // Create a texture with bad parameters and call a few methods.
    // Validation should fail but there should be not panic.
    initialize_test(TestParameters::default(), |ctx| {
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
    });
}

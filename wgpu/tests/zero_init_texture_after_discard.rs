use std::num::NonZeroU32;

use crate::common::{initialize_test, TestParameters};

const TEXTURE_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 64,
    height: 64,
    depth_or_array_layers: 1,
};
const BYTES_PER_PIXEL: u32 = 4;
const BUFFER_SIZE: u32 = TEXTURE_SIZE.width * TEXTURE_SIZE.height * BYTES_PER_PIXEL;
const BUFFER_COPY_LAYOUT: wgpu::ImageDataLayout = wgpu::ImageDataLayout {
    offset: 0,
    bytes_per_row: NonZeroU32::new(TEXTURE_SIZE.width * BYTES_PER_PIXEL),
    rows_per_image: None,
};

// Checks if discarding a color target resets its init state, causing a zero read of this texture when copied in after submit of the encoder.
#[test]
fn discarding_color_target_resets_texture_init_state_check_visible_on_copy_after_submit() {
    initialize_test(TestParameters::default(), |ctx| {
        let (texture, readback_buffer) = create_white_texture_and_readback_buffer(&ctx);
        {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color Discard"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: false, // discard!
                    },
                }],
                depth_stencil_attachment: None,
            }));
            ctx.queue.submit([encoder.finish()]);
        }
        {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            copy_texture_to_buffer(&mut encoder, &texture, &readback_buffer);
            ctx.queue.submit([encoder.finish()]);
        }
        assert_buffer_is_zero(&readback_buffer, &ctx.device);
    });
}

// Checks if discarding a color target resets its init state, causing a zero read of this texture when copied in the same encoder to a buffer.
#[test]
fn discarding_color_target_resets_texture_init_state_check_visible_on_copy_in_same_encoder() {
    initialize_test(TestParameters::default(), |ctx| {
        let (texture, readback_buffer) = create_white_texture_and_readback_buffer(&ctx);
        {
            let mut encoder = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            drop(encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Color Discard"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: false, // discard!
                    },
                }],
                depth_stencil_attachment: None,
            }));
            copy_texture_to_buffer(&mut encoder, &texture, &readback_buffer);
            ctx.queue.submit([encoder.finish()]);
        }
        assert_buffer_is_zero(&readback_buffer, &ctx.device);
    });
}

fn create_white_texture_and_readback_buffer(
    ctx: &crate::common::TestingContext,
) -> (wgpu::Texture, wgpu::Buffer) {
    // Size is chosen so that we don't need to care about buffer alignments.
    assert_eq!(
        (TEXTURE_SIZE.width * BYTES_PER_PIXEL) % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT,
        0
    );

    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("RenderTarget"),
        size: TEXTURE_SIZE,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::RENDER_ATTACHMENT,
    });

    // Clear using a write_texture operation. We could also clear using a render_pass clear.
    // However, when making this test intentionally fail (by breaking wgpu impl), it shows that at least on the tested Vulkan driver,
    // the later following discard pass in the test (i.e. internally vk::AttachmentStoreOp::DONT_CARE) will yield different depending on the operation we take here:
    // * clearing white -> discard will cause it to become black!
    // * clearing red -> discard will keep it red
    // * write_texture -> discard will keep buffer
    // This behavior is curious, but does not violate any spec - it is wgpu's job to pass this test no matter what a render target discard does.

    let data = vec![255; BUFFER_SIZE as usize];
    ctx.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        BUFFER_COPY_LAYOUT,
        TEXTURE_SIZE,
    );

    (
        texture,
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Texture Readback"),
            size: BUFFER_SIZE as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
    )
}

fn copy_texture_to_buffer(
    encoder: &mut wgpu::CommandEncoder,
    texture: &wgpu::Texture,
    read_back: &wgpu::Buffer,
) {
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: read_back,
            layout: BUFFER_COPY_LAYOUT,
        },
        TEXTURE_SIZE,
    );
}

fn assert_buffer_is_zero(readback_buffer: &wgpu::Buffer, device: &wgpu::Device) {
    {
        let buffer_slice = readback_buffer.slice(..);
        let _ = buffer_slice.map_async(wgpu::MapMode::Read);
        device.poll(wgpu::Maintain::Wait);
        let buffer_view = buffer_slice.get_mapped_range();

        assert!(
            buffer_view.iter().all(|b| *b == 0),
            "texture was not fully cleared"
        );
    }
    readback_buffer.unmap();
}

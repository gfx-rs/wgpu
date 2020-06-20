use std::env;
/// This example shows how to capture an image by rendering it to a texture, copying the texture to
/// a buffer, and retrieving it from the buffer. This could be used for "taking a screenshot," with
/// the added benefit that this method doesn't require a window to be created.
use std::fs::File;
use std::io::Write;
use std::mem::size_of;

async fn run() {
    let adapter = wgpu::Instance::new(wgpu::BackendBit::PRIMARY)
        .request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::Default,
                compatible_surface: None,
            },
            wgpu::UnsafeExtensions::disallow(),
        )
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                extensions: wgpu::Extensions::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    let args: Vec<_> = env::args().collect();
    let (width, height) = match args.len() {
        // 0 on wasm, 1 on desktop
        0 | 1 => (100usize, 200usize),
        3 => (args[1].parse().unwrap(), args[2].parse().unwrap()),
        _ => {
            println!("Incorrect number of arguments, possible usages:");
            println!("*   0 arguments - uses default width and height of (100, 200)");
            println!("*   2 arguments - uses specified width and height values");
            return;
        }
    };

    // It is a webgpu requirement that BufferCopyView.layout.bytes_per_row % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0
    // So we calculate padded_bytes_per_row by rounding unpadded_bytes_per_row
    // up to the next multiple of wgpu::COPY_BYTES_PER_ROW_ALIGNMENT.
    // https://en.wikipedia.org/wiki/Data_structure_alignment#Computing_padding
    let bytes_per_pixel = size_of::<u32>();
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let padded_bytes_per_row_padding = (align - unpadded_bytes_per_row % align) % align;
    let padded_bytes_per_row = unpadded_bytes_per_row + padded_bytes_per_row_padding;

    // The output buffer lets us retrieve the data as an array
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    let texture_extent = wgpu::Extent3d {
        width: width as u32,
        height: height as u32,
        depth: 1,
    };

    // The render pipeline renders data into this texture
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        size: texture_extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT | wgpu::TextureUsage::COPY_SRC,
        label: None,
    });

    // Set the background to be red
    let command_buffer = {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &texture.create_default_view(),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });

        // Copy the data from the texture to the buffer
        encoder.copy_texture_to_buffer(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::BufferCopyView {
                buffer: &output_buffer,
                layout: wgpu::TextureDataLayout {
                    offset: 0,
                    bytes_per_row: padded_bytes_per_row as u32,
                    rows_per_image: 0,
                },
            },
            texture_extent,
        );

        encoder.finish()
    };

    queue.submit(Some(command_buffer));

    // Note that we're not calling `.await` here.
    let buffer_slice = output_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    // If a file system is available, write the buffer as a PNG
    let has_file_system_available = cfg!(not(target_arch = "wasm32"));
    if !has_file_system_available {
        return;
    }

    if let Ok(()) = buffer_future.await {
        let padded_buffer = buffer_slice.get_mapped_range();

        let mut png_encoder = png::Encoder::new(
            File::create("red.png").unwrap(),
            width as u32,
            height as u32,
        );
        png_encoder.set_depth(png::BitDepth::Eight);
        png_encoder.set_color(png::ColorType::RGBA);
        let mut png_writer = png_encoder
            .write_header()
            .unwrap()
            .into_stream_writer_with_size(unpadded_bytes_per_row);

        // from the padded_buffer we write just the unpadded bytes into the image
        for chunk in padded_buffer.chunks(padded_bytes_per_row) {
            png_writer.write(&chunk[..unpadded_bytes_per_row]).unwrap();
        }
        png_writer.finish().unwrap();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(padded_buffer);

        output_buffer.unmap();
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        futures::executor::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}

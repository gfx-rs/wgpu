#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const TEXTURE_DIMS: (usize, usize) = (512, 512);

async fn run(path: Option<String>) {
    /* This will later store the raw pixel value data locally. We'll create it now as
    a convenient size reference. */
    let mut texture_data = Vec::<u8>::with_capacity(TEXTURE_DIMS.0 * TEXTURE_DIMS.1 * 4);

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default(), None)
        .await
        .unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let render_target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: TEXTURE_DIMS.0 as u32,
            height: TEXTURE_DIMS.1 as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[wgpu::TextureFormat::Rgba8UnormSrgb],
    });
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: texture_data.capacity() as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    /* let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[]
    }); */
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: None,
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::TextureFormat::Rgba8UnormSrgb.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    log::info!("Wgpu context set up.");

    //-----------------------------------------------

    let texture_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });
        render_pass.set_pipeline(&pipeline);
        render_pass.draw(0..3, 0..1);
    }
    // The texture now contains our rendered image
    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &render_target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                /* This needs to be a multiple of 256. Normally we would need to pad
                it but we here know it will work out anyways. */
                bytes_per_row: Some((TEXTURE_DIMS.0 * 4) as u32),
                rows_per_image: Some(TEXTURE_DIMS.1 as u32),
            },
        },
        wgpu::Extent3d {
            width: TEXTURE_DIMS.0 as u32,
            height: TEXTURE_DIMS.1 as u32,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(command_encoder.finish()));
    log::info!("Commands submitted.");

    //-----------------------------------------------

    // Time to get our image.
    let buffer_slice = output_staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap().unwrap();
    log::info!("Output buffer mapped.");
    {
        let view = buffer_slice.get_mapped_range();
        texture_data.extend_from_slice(&view[..]);
    }
    log::info!("Image data copied to local.");
    output_staging_buffer.unmap();

    #[cfg(not(target_arch = "wasm32"))]
    output_image_native(texture_data.to_vec(), path);
    #[cfg(target_arch = "wasm32")]
    output_image_wasm(texture_data.to_vec(), path);
    log::info!("Done.");
}

#[cfg(not(target_arch = "wasm32"))]
fn output_image_native(image_data: Vec<u8>, path: Option<String>) {
    let mut png_data = Vec::<u8>::with_capacity(image_data.len());
    let mut encoder = png::Encoder::new(
        std::io::Cursor::new(&mut png_data),
        TEXTURE_DIMS.0 as u32,
        TEXTURE_DIMS.1 as u32,
    );
    encoder.set_color(png::ColorType::Rgba);
    let mut png_writer = encoder.write_header().unwrap();
    png_writer.write_image_data(&image_data[..]).unwrap();
    png_writer.finish().unwrap();
    log::info!("Png file encoded in memory.");

    if let Some(p) = path {
        let mut file = std::fs::File::create(p).unwrap();
        file.write_all(&png_data[..]).unwrap();
    } else {
        log::warn!("No path specified. No file ultimately emitted.");
    }
}

#[cfg(target_arch = "wasm32")]
fn output_image_wasm(image_data: Vec<u8>, _path: Option<String>) {
    let document = web_sys::window().unwrap().document().unwrap();
    let body = document.body().unwrap();
    let canvas = document
        .create_element("canvas")
        .unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .unwrap();
    // We don't want to show the canvas, we just want it to exist in the background.
    canvas.set_attribute("hidden", "true").unwrap();
    let image_dimension_strings = (TEXTURE_DIMS.0.to_string(), TEXTURE_DIMS.1.to_string());
    canvas
        .set_attribute("width", image_dimension_strings.0.as_str())
        .unwrap();
    canvas
        .set_attribute("height", image_dimension_strings.1.as_str())
        .unwrap();
    canvas.set_attribute("background-color", "red").unwrap();
    body.append_child(&canvas).unwrap();
    log::info!("Set up canvas.");
    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();
    let image_data = web_sys::ImageData::new_with_u8_clamped_array(
        wasm_bindgen::Clamped(&image_data),
        TEXTURE_DIMS.0 as u32,
    )
    .unwrap();
    context.put_image_data(&image_data, 0.0, 0.0).unwrap();
    log::info!("Put image data in canvas.");
    // The canvas is now the image we ultimately want. We can create a data url from it now.
    let data_url = canvas.to_data_url().unwrap();
    let image_element = document
        .create_element("img")
        .unwrap()
        .dyn_into::<web_sys::HtmlImageElement>()
        .unwrap();
    image_element.set_src(&data_url);
    body.append_child(&image_element).unwrap();
    log::info!("Created image element with data url.");
    body.set_inner_html(
        &(body.inner_html()
            + r#"<p>The above image is for you!
        You can drag it to your desktop to download.</p>"#),
    );
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .format_timestamp_nanos()
            .init();

        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "please_don't_git_push_me.png".to_string());
        pollster::block_on(run(Some(path)));
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init_with_level(log::Level::Info).expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run(None));
    }
}

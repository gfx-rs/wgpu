//! This example demonstrates the basic usage of storage textures for the purpose of
//! creating a digital image of the Mandelbrot set
//! (<https://en.wikipedia.org/wiki/Mandelbrot_set>).
//!
//! Storage textures work like normal textures but they operate similar to storage buffers
//! in that they can be written to. The issue is that as it stands, write-only is the
//! only valid access mode for storage textures in WGSL and although there is a WGPU feature
//! to allow for read-write access, this is unfortunately a native-only feature and thus
//! we won't be using it here. If we needed a reference texture, we would need to add a
//! second texture to act as a reference and attach that as well. Luckily, we don't need
//! to read anything in our shader except the dimensions of our texture, which we can
//! easily get via `textureDimensions`.
//!
//! A lot of things aren't explained here via comments. See hello-compute and
//! repeated-compute for code that is more thoroughly commented.

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

const TEXTURE_DIMS: (usize, usize) = (512, 512);

async fn run(path: Option<String>) {
    let mut texture_data = vec![0u8; TEXTURE_DIMS.0 * TEXTURE_DIMS.1 * 4];

    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: TEXTURE_DIMS.0 as u32,
            height: TEXTURE_DIMS.1 as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let storage_texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());
    let output_staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: std::mem::size_of_val(&texture_data[..]) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        }],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&storage_texture_view),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
    });

    log::info!("Wgpu context set up.");
    //----------------------------------------

    let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass =
            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.set_pipeline(&pipeline);
        compute_pass.dispatch_workgroups(TEXTURE_DIMS.0 as u32, TEXTURE_DIMS.1 as u32, 1);
    }
    command_encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &storage_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                // This needs to be padded to 256.
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

    let buffer_slice = output_staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| sender.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    receiver.receive().await.unwrap().unwrap();
    log::info!("Output buffer mapped");
    {
        let view = buffer_slice.get_mapped_range();
        texture_data.copy_from_slice(&view[..]);
    }
    log::info!("GPU data copied to local.");
    output_staging_buffer.unmap();

    #[cfg(not(target_arch = "wasm32"))]
    output_image_native(texture_data.to_vec(), path);
    #[cfg(target_arch = "wasm32")]
    output_image_wasm(texture_data.to_vec(), path);
    log::info!("Done.")
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

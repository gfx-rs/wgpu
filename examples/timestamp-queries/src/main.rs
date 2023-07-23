//! Sample demonstrating different kinds of timestamp queries.

use std::borrow::Cow;

use wgpu::util::DeviceExt;

// Queries:
// * encoder timestamp start
// * encoder timestamp end
// * render start
// * render in-between (optional)
// * render end
// * compute start
// * compute in-between (optional)
// * compute end
const NUM_QUERIES: usize = 8;

struct Queries {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
}

impl Queries {
    fn new(device: &wgpu::Device) -> Self {
        Queries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: NUM_QUERIES as _,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size: (std::mem::size_of::<u64>() * NUM_QUERIES) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size: (std::mem::size_of::<u64>() * NUM_QUERIES) as wgpu::BufferAddress,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(&self.set, 0..NUM_QUERIES as u32, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    fn wait_and_print_results(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::Wait);

        {
            let timestamp_view = self
                .destination_buffer
                .slice(..(std::mem::size_of::<u64>() * NUM_QUERIES) as wgpu::BufferAddress)
                .get_mapped_range();

            let timestamps: &[u64] = bytemuck::cast_slice(&timestamp_view);
            println!("Raw timestamp buffer contents: {:?}", timestamps);

            let elapsed_us = |start, end: u64| {
                let period = queue.get_timestamp_period();
                end.wrapping_sub(start) as f64 * period as f64 / 1000.0
            };

            println!(
                "Elapsed time render + compute: {:.2} μs",
                elapsed_us(timestamps[0], timestamps[1])
            );

            println!(
                "Elapsed time render pass: {:.2} μs",
                elapsed_us(timestamps[2], timestamps[4])
            );
            if device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            {
                println!(
                    "Elapsed time first triangle: {:.2} μs",
                    elapsed_us(timestamps[3], timestamps[4])
                );
            }

            println!(
                "Elapsed time compute pass: {:.2} μs",
                elapsed_us(timestamps[5], timestamps[7])
            );
            if device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
            {
                println!(
                    "Elapsed time first compute: {:.2} μs",
                    elapsed_us(timestamps[5], timestamps[6])
                );
            }
        }

        self.destination_buffer.unmap();
    }
}

async fn run() {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::default();

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to request adapter.");

    // Check timestamp features.
    if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        println!("Adapter supports timestamp queries.");
    } else {
        println!("Adapter does not support timestamp queries, aborting.");
        return;
    }
    let mut features = wgpu::Features::empty() | wgpu::Features::TIMESTAMP_QUERY;
    if adapter
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        println!("Adapter supports timestamp queries within passes.");
        features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
    } else {
        println!("Adapter does not support timestamp queries within passes.");
    }

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features,
                limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let queries = Queries::new(&device);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    encoder.write_timestamp(&queries.set, 0);

    // Render two triangles and profile it.
    render_pass(&device, &shader, &mut encoder, &queries.set, 2);

    // Compute a hash function on a single thread a bunch of time and profile it.
    compute_pass(&device, &shader, &mut encoder, &queries.set, 5);

    encoder.write_timestamp(&queries.set, 1);

    queries.resolve(&mut encoder);
    queue.submit(Some(encoder.finish()));
    queries.wait_and_print_results(&device, &queue);
}

fn compute_pass(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    encoder: &mut wgpu::CommandEncoder,
    query_set: &wgpu::QuerySet,
    query_offset: u32,
) {
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&[42]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: "main_cs",
    });
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: storage_buffer.as_entire_binding(),
        }],
    });

    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(query_offset + 0),
            end_of_pass_write_index: Some(query_offset + 2),
        }),
    });
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.dispatch_workgroups(1, 1, 1);
    if device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        cpass.write_timestamp(query_set, query_offset + 1);
    }
    cpass.dispatch_workgroups(1, 1, 1);
}

fn render_pass(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    encoder: &mut wgpu::CommandEncoder,
    query_set: &wgpu::QuerySet,
    query_offset: u32,
) {
    let format = wgpu::TextureFormat::Rgba8Unorm;

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(format.into())],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    let render_target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("rendertarget"),
        size: wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[format],
    });
    let render_target_view = render_target.create_view(&wgpu::TextureViewDescriptor::default());

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &render_target_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::GREEN),
                store: true,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(query_offset + 0),
            end_of_pass_write_index: Some(query_offset + 2),
        }),
    });

    rpass.set_pipeline(&render_pipeline);

    rpass.draw(0..3, 0..1);
    if device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        rpass.write_timestamp(query_set, query_offset + 1);
    }

    rpass.draw(0..3, 0..1);
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        pollster::block_on(run());
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}

//! Sample demonstrating different kinds of gpu timestamp queries.
//!
//! Timestamp queries are typically used to profile how long certain operations take on the GPU.
//! wgpu has several ways of performing gpu timestamp queries:
//! * `wgpu::Encoder::write_timestamp` writes a between any commands recorded on an encoder.
//!     (enabled with wgpu::Features::TIMESTAMP_QUERY)
//! * passing `wgpu::RenderPassTimestampWrites`/`wgpu::ComputePassTimestampWrites` during render/compute pass creation.
//!     This writes timestamps for the beginning and end of a given pass.
//!     (enabled with wgpu::Features::TIMESTAMP_QUERY)
//! * `wgpu::RenderPass/ComputePass::write_timestamp` writes a timestamp within commands of a render pass.
//!     Note that some GPU architectures do not support this.
//!     (native only, enabled with wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
//!
//! Any timestamp is written to a `wgpu::QuerySet` which needs to be resolved to a buffer with `wgpu::BufferUsages::QUERY_RESOLVE`.
//! Since this usage is incompatible with `wgpu::BufferUsages::MAP_READ` we need to copy the resolved timestamps to a separate buffer afterwards.
//!
//! The period, i.e. the unit of time, of the timestamps in wgpu is undetermined and needs to be queried with `wgpu::Queue::get_timestamp_period`
//! in order to get comparable results.

use wgpu::util::DeviceExt;

struct Queries {
    set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
    num_queries: u64,
    next_unused_query: u32,
}

struct QueryResults {
    encoder_timestamps: [u64; 2],
    render_start_end_timestamps: [u64; 2],
    render_inside_timestamp: Option<u64>,
    compute_start_end_timestamps: [u64; 2],
    compute_inside_timestamp: Option<u64>,
}

impl QueryResults {
    // Queries:
    // * encoder timestamp start
    // * encoder timestamp end
    // * render start
    // * render in-between (optional)
    // * render end
    // * compute start
    // * compute in-between (optional)
    // * compute end
    const NUM_QUERIES: u64 = 8;

    #[allow(clippy::redundant_closure)] // False positive
    fn from_raw_results(timestamps: Vec<u64>, timestamps_inside_passes: bool) -> Self {
        assert_eq!(timestamps.len(), Self::NUM_QUERIES as usize);

        let mut next_slot = 0;
        let mut get_next_slot = || {
            let slot = timestamps[next_slot];
            next_slot += 1;
            slot
        };

        let mut encoder_timestamps = [0, 0];
        encoder_timestamps[0] = get_next_slot();
        let render_start_end_timestamps = [get_next_slot(), get_next_slot()];
        let render_inside_timestamp = timestamps_inside_passes.then(|| get_next_slot());
        let compute_start_end_timestamps = [get_next_slot(), get_next_slot()];
        let compute_inside_timestamp = timestamps_inside_passes.then(|| get_next_slot());
        encoder_timestamps[1] = get_next_slot();

        QueryResults {
            encoder_timestamps,
            render_start_end_timestamps,
            render_inside_timestamp,
            compute_start_end_timestamps,
            compute_inside_timestamp,
        }
    }

    #[cfg_attr(test, allow(unused))]
    fn print(&self, queue: &wgpu::Queue) {
        let period = queue.get_timestamp_period();
        let elapsed_us = |start, end: u64| end.wrapping_sub(start) as f64 * period as f64 / 1000.0;

        println!(
            "Elapsed time before render until after compute: {:.2} μs",
            elapsed_us(self.encoder_timestamps[0], self.encoder_timestamps[1]),
        );
        println!(
            "Elapsed time render pass: {:.2} μs",
            elapsed_us(
                self.render_start_end_timestamps[0],
                self.render_start_end_timestamps[1]
            )
        );
        if let Some(timestamp) = self.render_inside_timestamp {
            println!(
                "Elapsed time first triangle: {:.2} μs",
                elapsed_us(self.render_start_end_timestamps[0], timestamp)
            );
        }
        println!(
            "Elapsed time compute pass: {:.2} μs",
            elapsed_us(
                self.compute_start_end_timestamps[0],
                self.compute_start_end_timestamps[1]
            )
        );
        if let Some(timestamp) = self.compute_inside_timestamp {
            println!(
                "Elapsed time after first dispatch: {:.2} μs",
                elapsed_us(self.compute_start_end_timestamps[0], timestamp)
            );
        }
    }
}

impl Queries {
    fn new(device: &wgpu::Device, num_queries: u64) -> Self {
        Queries {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("Timestamp query set"),
                count: num_queries as _,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query resolve buffer"),
                size: std::mem::size_of::<u64>() as u64 * num_queries,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query dest buffer"),
                size: std::mem::size_of::<u64>() as u64 * num_queries,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
            num_queries,
            next_unused_query: 0,
        }
    }

    fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.resolve_query_set(
            &self.set,
            // TODO(https://github.com/gfx-rs/wgpu/issues/3993): Musn't be larger than the number valid queries in the set.
            0..self.next_unused_query,
            &self.resolve_buffer,
            0,
        );
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            self.resolve_buffer.size(),
        );
    }

    fn wait_for_results(&self, device: &wgpu::Device) -> Vec<u64> {
        self.destination_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |_| ());
        device.poll(wgpu::Maintain::wait()).panic_on_timeout();

        let timestamps = {
            let timestamp_view = self
                .destination_buffer
                .slice(..(std::mem::size_of::<u64>() as wgpu::BufferAddress * self.num_queries))
                .get_mapped_range();
            bytemuck::cast_slice(&timestamp_view).to_vec()
        };

        self.destination_buffer.unmap();

        timestamps
    }
}

#[cfg_attr(test, allow(unused))]
async fn run() {
    // Instantiates instance of wgpu
    let backends = wgpu::util::backend_bits_from_env().unwrap_or_default();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::from_build_config().with_env(),
        dx12_shader_compiler: wgpu::Dx12Compiler::default(),
        gles_minor_version: wgpu::Gles3MinorVersion::default(),
    });

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to request adapter.");

    // Check timestamp features.
    let features = adapter.features()
        & (wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    if features.contains(wgpu::Features::TIMESTAMP_QUERY) {
        println!("Adapter supports timestamp queries.");
    } else {
        println!("Adapter does not support timestamp queries, aborting.");
        return;
    }
    let timestamps_inside_passes = features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    if timestamps_inside_passes {
        println!("Adapter supports timestamp queries within passes.");
    } else {
        println!("Adapter does not support timestamp queries within passes.");
    }

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        )
        .await
        .unwrap();

    let queries = submit_render_and_compute_pass_with_queries(&device, &queue);
    let raw_results = queries.wait_for_results(&device);
    println!("Raw timestamp buffer contents: {:?}", raw_results);
    QueryResults::from_raw_results(raw_results, timestamps_inside_passes).print(&queue);
}

fn submit_render_and_compute_pass_with_queries(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Queries {
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut queries = Queries::new(device, QueryResults::NUM_QUERIES);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!("shader.wgsl"))),
    });

    encoder.write_timestamp(&queries.set, queries.next_unused_query);
    queries.next_unused_query += 1;

    // Render two triangles and profile it.
    render_pass(
        device,
        &shader,
        &mut encoder,
        &queries.set,
        &mut queries.next_unused_query,
    );

    // Compute a hash function on a single thread a bunch of time and profile it.
    compute_pass(
        device,
        &shader,
        &mut encoder,
        &queries.set,
        &mut queries.next_unused_query,
    );

    encoder.write_timestamp(&queries.set, queries.next_unused_query);
    queries.next_unused_query += 1;

    queries.resolve(&mut encoder);
    queue.submit(Some(encoder.finish()));

    queries
}

fn compute_pass(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    encoder: &mut wgpu::CommandEncoder,
    query_set: &wgpu::QuerySet,
    next_unused_query: &mut u32,
) {
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&[42]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module,
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
            beginning_of_pass_write_index: Some(*next_unused_query),
            end_of_pass_write_index: Some(*next_unused_query + 1),
        }),
    });
    *next_unused_query += 2;
    cpass.set_pipeline(&compute_pipeline);
    cpass.set_bind_group(0, &bind_group, &[]);
    cpass.dispatch_workgroups(1, 1, 1);
    if device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        cpass.write_timestamp(query_set, *next_unused_query);
        *next_unused_query += 1;
    }
    cpass.dispatch_workgroups(1, 1, 1);
}

fn render_pass(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    encoder: &mut wgpu::CommandEncoder,
    query_set: &wgpu::QuerySet,
    next_unused_query: &mut u32,
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
            module,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
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
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(*next_unused_query),
            end_of_pass_write_index: Some(*next_unused_query + 1),
        }),
        occlusion_query_set: None,
    });
    *next_unused_query += 2;

    rpass.set_pipeline(&render_pipeline);

    rpass.draw(0..3, 0..1);
    if device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES)
    {
        rpass.write_timestamp(query_set, *next_unused_query);
        *next_unused_query += 1;
    }

    rpass.draw(0..3, 0..1);
}

pub fn main() {
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

#[cfg(test)]
mod tests {
    use wgpu_test::{gpu_test, GpuTestConfiguration};

    use super::{submit_render_and_compute_pass_with_queries, QueryResults};

    #[gpu_test]
    static TIMESTAMPS_ENCODER: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(
            wgpu_test::TestParameters::default()
                .limits(wgpu::Limits::downlevel_defaults())
                .features(wgpu::Features::TIMESTAMP_QUERY),
        )
        .run_sync(|ctx| test_timestamps(ctx, false));

    #[gpu_test]
    static TIMESTAMPS_PASSES: GpuTestConfiguration = GpuTestConfiguration::new()
        .parameters(
            wgpu_test::TestParameters::default()
                .limits(wgpu::Limits::downlevel_defaults())
                .features(
                    wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
                ),
        )
        .run_sync(|ctx| test_timestamps(ctx, true));

    fn test_timestamps(ctx: wgpu_test::TestingContext, timestamps_inside_passes: bool) {
        let queries = submit_render_and_compute_pass_with_queries(&ctx.device, &ctx.queue);
        let raw_results = queries.wait_for_results(&ctx.device);
        let QueryResults {
            encoder_timestamps,
            render_start_end_timestamps,
            render_inside_timestamp,
            compute_start_end_timestamps,
            compute_inside_timestamp,
        } = QueryResults::from_raw_results(raw_results, timestamps_inside_passes);

        // Timestamps may wrap around, so can't really only reason about deltas!
        // Making things worse, deltas are allowed to be zero.
        let render_delta =
            render_start_end_timestamps[1].wrapping_sub(render_start_end_timestamps[0]);
        let compute_delta =
            compute_start_end_timestamps[1].wrapping_sub(compute_start_end_timestamps[0]);
        let encoder_delta = encoder_timestamps[1].wrapping_sub(encoder_timestamps[0]);

        assert!(encoder_delta > 0);
        assert!(encoder_delta >= render_delta + compute_delta);

        if let Some(render_inside_timestamp) = render_inside_timestamp {
            assert!(render_inside_timestamp >= render_start_end_timestamps[0]);
            assert!(render_inside_timestamp <= render_start_end_timestamps[1]);
        }
        if let Some(compute_inside_timestamp) = compute_inside_timestamp {
            assert!(compute_inside_timestamp >= compute_start_end_timestamps[0]);
            assert!(compute_inside_timestamp <= compute_start_end_timestamps[1]);
        }
    }
}

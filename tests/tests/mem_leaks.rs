#[cfg(any(
    not(target_arch = "wasm32"),
    target_os = "emscripten",
    feature = "webgl"
))]
async fn draw_test_with_reports(
    ctx: wgpu_test::TestingContext,
    expected: &[u32],
    function: impl FnOnce(&mut wgpu::RenderPass<'_>),
) {
    use std::num::NonZeroU64;

    use wgpu::util::DeviceExt;

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.devices.num_allocated, 1);
    assert_eq!(report.queues.num_allocated, 1);

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("./vertex_indices/draw.vert.wgsl"));

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.shader_modules.num_allocated, 1);

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(4),
                },
                visibility: wgpu::ShaderStages::VERTEX,
                count: None,
            }],
        });

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 0);
    assert_eq!(report.bind_groups.num_allocated, 0);
    assert_eq!(report.bind_group_layouts.num_allocated, 1);

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4 * expected.len() as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);
    assert_eq!(report.bind_groups.num_allocated, 1);
    assert_eq!(report.bind_group_layouts.num_allocated, 1);

    let ppl = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);
    assert_eq!(report.pipeline_layouts.num_allocated, 1);
    assert_eq!(report.render_pipelines.num_allocated, 0);
    assert_eq!(report.compute_pipelines.num_allocated, 0);

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&ppl),
            vertex: wgpu::VertexState {
                buffers: &[],
                module: &shader,
                entry_point: Some("vs_main_builtin"),
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);
    assert_eq!(report.bind_groups.num_allocated, 1);
    assert_eq!(report.bind_group_layouts.num_allocated, 1);
    assert_eq!(report.shader_modules.num_allocated, 1);
    assert_eq!(report.pipeline_layouts.num_allocated, 1);
    assert_eq!(report.render_pipelines.num_allocated, 1);
    assert_eq!(report.compute_pipelines.num_allocated, 0);

    drop(shader);

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.shader_modules.num_allocated, 0);
    assert_eq!(report.shader_modules.num_kept_from_user, 0);
    assert_eq!(report.textures.num_allocated, 0);
    assert_eq!(report.texture_views.num_allocated, 0);

    let texture = ctx.device.create_texture_with_data(
        &ctx.queue,
        &wgpu::TextureDescriptor {
            label: Some("dummy"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[0, 0, 0, 1],
    );
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);
    assert_eq!(report.texture_views.num_allocated, 1);
    assert_eq!(report.textures.num_allocated, 1);

    drop(texture);

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);
    assert_eq!(report.texture_views.num_allocated, 1);
    assert_eq!(report.texture_views.num_kept_from_user, 1);
    assert_eq!(report.textures.num_allocated, 0);
    assert_eq!(report.textures.num_kept_from_user, 0);

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.command_buffers.num_allocated, 1);
    assert_eq!(report.buffers.num_allocated, 1);

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: None,
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            ops: wgpu::Operations::default(),
            resolve_target: None,
            view: &texture_view,
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    rpass.set_pipeline(&pipeline);
    rpass.set_bind_group(0, &bg, &[]);

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.buffers.num_allocated, 1);
    assert_eq!(report.bind_groups.num_allocated, 1);
    assert_eq!(report.bind_group_layouts.num_allocated, 1);
    assert_eq!(report.pipeline_layouts.num_allocated, 1);
    assert_eq!(report.render_pipelines.num_allocated, 1);
    assert_eq!(report.compute_pipelines.num_allocated, 0);
    assert_eq!(report.command_buffers.num_allocated, 1);
    assert_eq!(report.render_bundles.num_allocated, 0);
    assert_eq!(report.texture_views.num_allocated, 1);
    assert_eq!(report.textures.num_allocated, 0);

    function(&mut rpass);

    drop(rpass);
    drop(pipeline);
    drop(texture_view);
    drop(ppl);
    drop(bgl);
    drop(bg);
    drop(buffer);

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();
    assert_eq!(report.command_buffers.num_kept_from_user, 1);
    assert_eq!(report.render_pipelines.num_kept_from_user, 0);
    assert_eq!(report.pipeline_layouts.num_kept_from_user, 0);
    assert_eq!(report.bind_group_layouts.num_kept_from_user, 0);
    assert_eq!(report.bind_groups.num_kept_from_user, 0);
    assert_eq!(report.buffers.num_kept_from_user, 0);
    assert_eq!(report.texture_views.num_kept_from_user, 0);
    assert_eq!(report.textures.num_kept_from_user, 0);
    assert_eq!(report.command_buffers.num_allocated, 1);
    assert_eq!(report.render_pipelines.num_allocated, 0);
    assert_eq!(report.pipeline_layouts.num_allocated, 0);
    assert_eq!(report.bind_group_layouts.num_allocated, 0);
    assert_eq!(report.bind_groups.num_allocated, 0);
    assert_eq!(report.buffers.num_allocated, 0);
    assert_eq!(report.texture_views.num_allocated, 0);
    assert_eq!(report.textures.num_allocated, 0);

    let submit_index = ctx.queue.submit(Some(encoder.finish()));

    // TODO: fix in https://github.com/gfx-rs/wgpu/pull/5141
    // let global_report = ctx.instance.generate_report().unwrap();
    // let report = global_report.hub_report();
    // assert_eq!(report.command_buffers.num_allocated, 0);

    ctx.async_poll(wgpu::Maintain::wait_for(submit_index))
        .await
        .panic_on_timeout();

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();

    assert_eq!(report.render_pipelines.num_allocated, 0);
    assert_eq!(report.bind_groups.num_allocated, 0);
    assert_eq!(report.bind_group_layouts.num_allocated, 0);
    assert_eq!(report.pipeline_layouts.num_allocated, 0);
    assert_eq!(report.texture_views.num_allocated, 0);
    assert_eq!(report.textures.num_allocated, 0);
    assert_eq!(report.buffers.num_allocated, 0);

    drop(ctx.queue);
    drop(ctx.device);
    drop(ctx.adapter);

    let global_report = ctx.instance.generate_report().unwrap();
    let report = global_report.hub_report();

    assert_eq!(report.queues.num_kept_from_user, 0);
    assert_eq!(report.textures.num_kept_from_user, 0);
    assert_eq!(report.devices.num_kept_from_user, 0);
    assert_eq!(report.queues.num_allocated, 0);
    assert_eq!(report.buffers.num_allocated, 0);
    assert_eq!(report.textures.num_allocated, 0);
    assert_eq!(report.texture_views.num_allocated, 0);
    assert_eq!(report.devices.num_allocated, 0);
}

#[cfg(any(
    not(target_arch = "wasm32"),
    target_os = "emscripten",
    feature = "webgl"
))]
#[wgpu_test::gpu_test]
static SIMPLE_DRAW_CHECK_MEM_LEAKS: wgpu_test::GpuTestConfiguration =
    wgpu_test::GpuTestConfiguration::new()
        .parameters(
            wgpu_test::TestParameters::default()
                .test_features_limits()
                .features(wgpu::Features::VERTEX_WRITABLE_STORAGE),
        )
        .run_async(|ctx| {
            draw_test_with_reports(ctx, &[0, 1, 2, 3, 4, 5], |cmb| {
                cmb.draw(0..6, 0..1);
            })
        });

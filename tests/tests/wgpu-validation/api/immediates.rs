//! Validation tests for `var<immediate>`

use wgpu_test::fail;

const COMPUTE_SHADER: &str = "
    var<immediate> im: vec4<f32>;

    @group(0) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn main() {
        output = im;
    }
";

fn setup_compute() -> (
    wgpu::Device,
    wgpu::Queue,
    wgpu::ComputePipeline,
    wgpu::BindGroup,
) {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 16,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    (device, queue, pipeline, bind_group)
}

const SHADER_RENDER: &str = "
    const POSITION: vec4f = vec4f(0, 0, 0, 1);

    var<immediate> im: vec4<f32>;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) index: u32,
    }

    @vertex fn vertex(
        @builtin(vertex_index) ix: u32,
    ) -> VertexOutput {
        return VertexOutput(POSITION, ix);
    }

    @fragment fn fragment(
        @location(0) @interpolate(flat) ix: u32,
     ) -> @location(0) vec4f {
        let fc = im[ix];
        return vec4f(0, 0, 0, fc);
    }
";

fn setup_render() -> (
    wgpu::Device,
    wgpu::Queue,
    wgpu::RenderPipeline,
    wgpu::TextureView,
) {
    let (device, queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let output_texture = device.create_texture(&wgpu::TextureDescriptor {
        size: wgpu::Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        label: Some("Output Texture"),
        view_formats: &[],
    });
    let output_texture_view = output_texture.create_view(&Default::default());

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_RENDER.into()),
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 8 * size_of::<u32>() as u32,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: None,
            buffers: &[],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: None,
            targets: &[Some(output_texture.format().into())],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::PointList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    (device, queue, pipeline, output_texture_view)
}

fn begin_render_pass<'b, 'a: 'b>(
    output_texture_view: &'a wgpu::TextureView,
    encoder: &'a mut wgpu::CommandEncoder,
    f: impl Fn(&mut wgpu::RenderPass<'b>),
) {
    let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: output_texture_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::default()),
                store: wgpu::StoreOp::Store,
            },
        })],
        ..Default::default()
    });
    f(&mut render_pass);
}

#[test]
fn dispatch_without_setting_immediates_fails() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fail(&device, || encoder.finish(), Some("immediate data"));
}

#[test]
fn dispatch_with_partial_immediates_fails() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 8]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    fail(&device, || encoder.finish(), Some("immediate data"));
}

#[test]
fn dispatch_with_all_immediates_set_succeeds() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn dispatch_with_incremental_immediates_succeeds() {
    let (device, _queue, pipeline, bind_group) = setup_compute();

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 8]);
        pass.set_immediates(8, &[0u8; 8]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn render_without_setting_immediates_fails() {
    let (device, _queue, pipeline, output_texture_view) = setup_render();

    fn do_encoding<'a>(
        encoder: &mut dyn wgpu::util::RenderEncoder<'a>,
        pipeline: &'a wgpu::RenderPipeline,
    ) {
        encoder.set_pipeline(&pipeline);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        do_encoding(pass, &pipeline);
    });

    fail(&device, || encoder.finish(), Some("immediate data"));

    let mut bundle_encoder =
        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..wgpu::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);

    fail(
        &device,
        || bundle_encoder.finish(&wgpu::RenderBundleDescriptor::default()),
        Some("immediate data"),
    );
}

#[test]
fn render_with_partial_immediates_fails() {
    let (device, _queue, pipeline, output_texture_view) = setup_render();

    fn do_encoding<'a>(
        encoder: &mut dyn wgpu::util::RenderEncoder<'a>,
        pipeline: &'a wgpu::RenderPipeline,
    ) {
        encoder.set_pipeline(&pipeline);
        encoder.set_immediates(0, &[0u8; 8]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        do_encoding(pass, &pipeline);
    });

    fail(&device, || encoder.finish(), Some("immediate data"));

    let mut bundle_encoder =
        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..wgpu::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);

    fail(
        &device,
        || bundle_encoder.finish(&wgpu::RenderBundleDescriptor::default()),
        Some("immediate data"),
    );
}

#[test]
fn render_with_all_immediates_set_succeeds() {
    let (device, _queue, pipeline, output_texture_view) = setup_render();

    fn do_encoding<'a>(
        encoder: &mut dyn wgpu::util::RenderEncoder<'a>,
        pipeline: &'a wgpu::RenderPipeline,
    ) {
        encoder.set_pipeline(&pipeline);
        encoder.set_immediates(0, &[0u8; 16]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        do_encoding(pass, &pipeline);
    });

    wgpu_test::valid(&device, || encoder.finish());

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut bundle_encoder =
        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..wgpu::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);
    let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        pass.execute_bundles([&bundle]);
    });

    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn render_with_incremental_immediates_succeeds() {
    let (device, _queue, pipeline, output_texture_view) = setup_render();

    fn do_encoding<'a>(
        encoder: &mut dyn wgpu::util::RenderEncoder<'a>,
        pipeline: &'a wgpu::RenderPipeline,
    ) {
        encoder.set_pipeline(&pipeline);
        encoder.set_immediates(0, &[0u8; 8]);
        encoder.set_immediates(8, &[0u8; 8]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        do_encoding(pass, &pipeline);
    });

    wgpu_test::valid(&device, || encoder.finish());

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut bundle_encoder =
        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..wgpu::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);
    let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        pass.execute_bundles([&bundle]);
    });

    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn render_bundle_without_resetting_immediates_fails() {
    let (device, _queue, pipeline, output_texture_view) = setup_render();

    fn do_encoding<'a>(
        encoder: &mut dyn wgpu::util::RenderEncoder<'a>,
        pipeline: &'a wgpu::RenderPipeline,
    ) {
        encoder.set_pipeline(&pipeline);
        encoder.set_immediates(0, &[0u8; 16]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut bundle_encoder =
        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..wgpu::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);
    let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        pass.execute_bundles([&bundle]);
        pass.set_pipeline(&pipeline);
        pass.draw(0..4, 0..1);
    });

    fail(&device, || encoder.finish(), Some("immediate data"));
}

#[test]
fn render_bundle_with_resetting_immediates_succeeds() {
    let (device, _queue, pipeline, output_texture_view) = setup_render();

    fn do_encoding<'a>(
        encoder: &mut dyn wgpu::util::RenderEncoder<'a>,
        pipeline: &'a wgpu::RenderPipeline,
    ) {
        encoder.set_pipeline(&pipeline);
        encoder.set_immediates(0, &[0u8; 16]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = device.create_command_encoder(&Default::default());
    let mut bundle_encoder =
        device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            color_formats: &[Some(output_texture_view.texture().format())],
            sample_count: 1,
            ..wgpu::RenderBundleEncoderDescriptor::default()
        });
    do_encoding(&mut bundle_encoder, &pipeline);
    let bundle = bundle_encoder.finish(&wgpu::RenderBundleDescriptor::default());

    begin_render_pass(&output_texture_view, &mut encoder, |pass| {
        do_encoding(pass, &pipeline);
        pass.execute_bundles([&bundle]);
        do_encoding(pass, &pipeline);
    });

    wgpu_test::valid(&device, || encoder.finish());
}

const STRUCT_SHADER: &str = "
    struct S {
        a: f32,
        // 12 bytes padding
        b: vec4<f32>,
    }
    var<immediate> im: S;

    @group(0) @binding(0)
    var<storage, read_write> output: vec4<f32>;

    @compute @workgroup_size(1)
    fn main() {
        output = im.b;
    }
";

#[test]
fn struct_padding_slots_not_required() {
    let (device, _q) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(STRUCT_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 32,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        // skip padding at bytes 4..16
        pass.set_immediates(0, &[0u8; 4]);
        pass.set_immediates(16, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

const NO_IMMEDIATES_SHADER: &str = "
    @group(0) @binding(0)
    var<storage, read_write> output: u32;

    @compute @workgroup_size(1)
    fn main() {
        output = 42u;
    }
";

#[test]
fn pipeline_without_immediates_needs_none() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor::default());

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(NO_IMMEDIATES_SHADER.into()),
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&layout),
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

#[test]
fn auto_layout_infers_immediate_size() {
    let (device, _q) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::IMMEDIATES,
        required_limits: wgpu::Limits {
            max_immediate_size: 64,
            ..Default::default()
        },
        ..Default::default()
    });

    let sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &sm,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_immediates(0, &[0u8; 16]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    wgpu_test::valid(&device, || encoder.finish());
}

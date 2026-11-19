use std::num::NonZeroU64;

use wgpu::util::RenderEncoder;
use wgpu::*;
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        PARTIAL_UPDATE,
        RENDER_PASS_TEST,
        NESTED_STRUCT,
        BEYOND_LAYOUT_SIZE,
        LARGER_LAYOUT_AFTER_DRAW,
    ]);
}

const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

/// A storage buffer at `@group(0) @binding(0)` that shaders write and tests read back.
struct Output {
    bind_group_layout: BindGroupLayout,
    bind_group: BindGroup,
    buffer: Buffer,
}

impl Output {
    fn new(ctx: &TestingContext, visibility: ShaderStages, size: BufferAddress) -> Self {
        let bind_group_layout = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(size),
                    },
                    count: None,
                }],
            });
        let buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            bind_group_layout,
            bind_group,
            buffer,
        }
    }

    /// Submits `encoder` and returns the contents of the buffer.
    async fn read<T: bytemuck::Pod>(
        &self,
        ctx: &TestingContext,
        mut encoder: CommandEncoder,
    ) -> Vec<T> {
        let readback = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: self.buffer.size(),
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &readback, 0, self.buffer.size());
        ctx.queue.submit([encoder.finish()]);

        readback.slice(..).map_async(MapMode::Read, |_| ());
        ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();

        let data = readback.slice(..).get_mapped_range().unwrap();
        bytemuck::cast_slice::<u8, T>(&data).to_vec()
    }
}

fn create_pipeline_layout(
    ctx: &TestingContext,
    bind_group_layouts: &[Option<&BindGroupLayout>],
    immediate_size: u32,
) -> PipelineLayout {
    ctx.device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts,
            immediate_size,
        })
}

/// Creates a pipeline with the only compute entry point of `module`.
fn create_compute_pipeline(
    ctx: &TestingContext,
    module: &ShaderModule,
    layout: &PipelineLayout,
) -> ComputePipeline {
    ctx.device
        .create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(layout),
            module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        })
}

/// Creates a pipeline that draws points with the only vertex entry point of `module`.
fn create_point_pipeline(
    ctx: &TestingContext,
    module: &ShaderModule,
    layout: &PipelineLayout,
    fs_entry_point: &str,
) -> RenderPipeline {
    ctx.device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: VertexState {
                module,
                entry_point: None,
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module,
                entry_point: Some(fs_entry_point),
                targets: &[Some(TARGET_FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
}

/// Creates a 1x1 render target.
fn create_target(ctx: &TestingContext) -> TextureView {
    ctx.device
        .create_texture(&TextureDescriptor {
            label: None,
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
            texture_binding_view_dimension: None,
        })
        .create_view(&Default::default())
}

fn begin_render_pass<'a>(encoder: &'a mut CommandEncoder, view: &TextureView) -> RenderPass<'a> {
    encoder.begin_render_pass(&RenderPassDescriptor {
        color_attachments: &[Some(RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::default()),
                store: StoreOp::Store,
            },
        })],
        ..Default::default()
    })
}

fn create_bundle_encoder(ctx: &TestingContext) -> RenderBundleEncoder<'_> {
    ctx.device
        .create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
            color_formats: &[Some(TARGET_FORMAT)],
            sample_count: 1,
            ..Default::default()
        })
}

fn create_command_encoder(ctx: &TestingContext) -> CommandEncoder {
    ctx.device
        .create_command_encoder(&CommandEncoderDescriptor::default())
}

fn create_wgsl_module(ctx: &TestingContext, source: &str) -> ShaderModule {
    ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(source.into()),
    })
}

/// We want to test that partial updates to immediates work as expected.
///
/// As such, we dispatch two compute passes, one which writes the values
/// before a partial update, and one which writes the values after the partial update.
///
/// If the update code is working correctly, the values not written to by the second update
/// will remain unchanged.
#[apply(gpu_test!)]
static PARTIAL_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::IMMEDIATES)
            .limits(wgpu::Limits {
                max_immediate_size: 32,
                ..Default::default()
            }),
    )
    .run_async(partial_update_test);

const SHADER: &str = r#"
    struct Pc {
        offset: u32,
        vector: vec4f,
    }

    var<immediate> pc: Pc;

    @group(0) @binding(0)
    var<storage, read_write> output: array<vec4f>;

    @compute @workgroup_size(1)
    fn main() {
        output[pc.offset] = pc.vector;
    }
"#;

async fn partial_update_test(ctx: TestingContext) {
    let module = create_wgsl_module(&ctx, SHADER);
    let output = Output::new(&ctx, ShaderStages::COMPUTE, 32);
    let layout = create_pipeline_layout(&ctx, &[Some(&output.bind_group_layout)], 32);
    let pipeline = create_compute_pipeline(&ctx, &module, &layout);

    let mut encoder = create_command_encoder(&ctx);
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &output.bind_group, &[]);

        // -- Dispatch 0 --

        // Dispatch number
        cpass.set_immediates(0, bytemuck::bytes_of(&[0_u32]));
        // Update the whole vector.
        cpass.set_immediates(16, bytemuck::bytes_of(&[1.0_f32, 2.0, 3.0, 4.0]));
        cpass.dispatch_workgroups(1, 1, 1);

        // -- Dispatch 1 --

        // Dispatch number
        cpass.set_immediates(0, bytemuck::bytes_of(&[1_u32]));
        // Update just the y component of the vector.
        cpass.set_immediates(20, bytemuck::bytes_of(&[5.0_f32]));
        cpass.dispatch_workgroups(1, 1, 1);
    }

    let floats = output.read::<f32>(&ctx, encoder).await;

    // first 4 floats the initial value
    // second 4 floats the first update
    assert_eq!(floats, [1.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 4.0]);
}

#[apply(gpu_test!)]
static RENDER_PASS_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::IMMEDIATES | Features::VERTEX_WRITABLE_STORAGE)
            .limits(wgpu::Limits {
                max_immediate_size: 64,
                ..Default::default()
            }),
    )
    .run_async(move |ctx| async move {
        for use_render_bundle in [false, true] {
            render_pass_test(&ctx, use_render_bundle).await;
        }
    });

// This shader simply  moves the values from vector_constants and immediates into the
// result buffer.  It expects to be called 4 times (with vector_index in 0..4) with its
// topology being PointList, so that each vertex shader call leads to exactly one fragment
// call.
const SHADER2: &str = "
    const POSITION: vec4f = vec4f(0, 0, 0, 1);

    struct ImmediateData {
        vertex_constants: vec4i,
        fragment_constants: vec4i,
    }

    var<immediate> immediates: ImmediateData;

    @group(0) @binding(0) var<storage, read_write> result: array<i32>;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) @interpolate(flat) index: u32,
    }

    @vertex fn vertex(
        @builtin(vertex_index) ix: u32,
    ) -> VertexOutput {
        result[ix] = immediates.vertex_constants[ix];
        return VertexOutput(POSITION, ix);
    }

    @fragment fn fragment(
        @location(0) @interpolate(flat) ix: u32,
     ) -> @location(0) vec4f {
        result[ix + 4u] = immediates.fragment_constants[ix];
        return vec4f();
    }
";

async fn render_pass_test(ctx: &TestingContext, use_render_bundle: bool) {
    // We need an output texture, even though we're not ever going to look at it.
    let view = create_target(ctx);
    let module = create_wgsl_module(ctx, SHADER2);
    let output = Output::new(ctx, ShaderStages::VERTEX_FRAGMENT, 32);
    let layout = create_pipeline_layout(ctx, &[Some(&output.bind_group_layout)], 32);
    let pipeline = create_point_pipeline(ctx, &module, &layout, "fragment");

    let data: Vec<i32> = (0..8).map(|i| (i * i) - 1).collect();

    fn do_encoding<'a>(
        encoder: &mut dyn RenderEncoder<'a>,
        pipeline: &'a RenderPipeline,
        bind_group: &'a BindGroup,
        data: &'a Vec<i32>,
    ) {
        let data_as_u8: &[u8] = bytemuck::cast_slice(data.as_slice());
        encoder.set_pipeline(pipeline);
        encoder.set_immediates(0, data_as_u8);
        encoder.set_bind_group(0, Some(bind_group), &[]);
        encoder.draw(0..4, 0..1);
    }

    let mut encoder = create_command_encoder(ctx);
    {
        let mut render_pass = begin_render_pass(&mut encoder, &view);
        if use_render_bundle {
            // Execute the commands in a render_bundle_encoder.
            let mut render_bundle_encoder = create_bundle_encoder(ctx);
            do_encoding(
                &mut render_bundle_encoder,
                &pipeline,
                &output.bind_group,
                &data,
            );
            let render_bundle = render_bundle_encoder.finish(&RenderBundleDescriptor::default());
            render_pass.execute_bundles([&render_bundle]);
        } else {
            // Execute the commands directly.
            do_encoding(&mut render_pass, &pipeline, &output.bind_group, &data);
        }
    }
    let result = output.read::<i32>(ctx, encoder).await;
    assert_eq!(&result, &data);
}

/// Immediates have the same layout as storage buffers, so a struct-typed member is not
/// aligned to 16 bytes: `inner` below lives at offset 4, directly after `x`.
///
/// Backends that read immediates through a constant buffer have to take this into account,
/// since for example HLSL starts every nested struct on a new 16-byte register.
#[apply(gpu_test!)]
static NESTED_STRUCT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::IMMEDIATES)
            .limits(Limits {
                max_immediate_size: 16,
                ..Default::default()
            }),
    )
    .run_async(nested_struct_test);

const NESTED_STRUCT_SHADER: &str = r#"
    struct Inner { a: u32, b: u32 }
    struct Data { x: u32, inner: Inner, y: u32 }

    var<immediate> data: Data;

    @group(0) @binding(0)
    var<storage, read_write> output: array<u32, 4>;

    @compute @workgroup_size(1)
    fn main() {
        output[0] = data.x;
        output[1] = data.inner.a;
        output[2] = data.inner.b;
        output[3] = data.y;
    }
"#;

async fn nested_struct_test(ctx: TestingContext) {
    let module = create_wgsl_module(&ctx, NESTED_STRUCT_SHADER);
    let output = Output::new(&ctx, ShaderStages::COMPUTE, 16);
    let layout = create_pipeline_layout(&ctx, &[Some(&output.bind_group_layout)], 16);
    let pipeline = create_compute_pipeline(&ctx, &module, &layout);

    let mut encoder = create_command_encoder(&ctx);
    {
        let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &output.bind_group, &[]);
        // `Data { x: 0, inner: Inner { a: 1, b: 2 }, y: 3 }`
        cpass.set_immediates(0, bytemuck::bytes_of(&[0_u32, 1, 2, 3]));
        cpass.dispatch_workgroups(1, 1, 1);
    }
    assert_eq!(output.read::<u32>(&ctx, encoder).await, [0, 1, 2, 3]);
}

/// Shader for tests that write more immediates than a pipeline layout holds.
///
/// `fs_write` copies the immediates to `output`. The other entry points use nothing.
const LAYOUT_SIZE_SHADER: &str = r#"
    var<immediate> data: vec2u;

    @group(0) @binding(0)
    var<storage, read_write> output: vec2u;

    @compute @workgroup_size(1)
    fn cs_main() {}

    @vertex
    fn vs_main() -> @builtin(position) vec4f {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    @fragment
    fn fs_empty() -> @location(0) vec4f {
        return vec4f();
    }

    @fragment
    fn fs_write() -> @location(0) vec4f {
        output = data;
        return vec4f();
    }
"#;

/// `set_immediates` is limited by `max_immediate_size`, not by the pipeline layout.
///
/// Write immediates that are empty or larger than the pipeline layout immediate size,
/// including layouts with no immediates, then dispatch and draw.
#[apply(gpu_test!)]
static BEYOND_LAYOUT_SIZE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::IMMEDIATES)
            .limits(Limits {
                max_immediate_size: 16,
                ..Default::default()
            }),
    )
    .run_async(beyond_layout_size_test);

async fn beyond_layout_size_test(ctx: TestingContext) {
    let module = create_wgsl_module(&ctx, LAYOUT_SIZE_SHADER);
    let view = create_target(&ctx);
    let data = [0_u8; 16];

    for immediate_size in [0, 4] {
        let layout = create_pipeline_layout(&ctx, &[], immediate_size);
        let compute_pipeline = create_compute_pipeline(&ctx, &module, &layout);
        let render_pipeline = create_point_pipeline(&ctx, &module, &layout, "fs_empty");

        for data in [&data[..0], &data[..]] {
            let mut bundle_encoder = create_bundle_encoder(&ctx);
            bundle_encoder.set_pipeline(&render_pipeline);
            bundle_encoder.set_immediates(0, data);
            bundle_encoder.draw(0..1, 0..1);
            let bundle = bundle_encoder.finish(&RenderBundleDescriptor::default());

            let mut encoder = create_command_encoder(&ctx);
            {
                let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor::default());
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_immediates(0, data);
                cpass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut rpass = begin_render_pass(&mut encoder, &view);
                rpass.set_pipeline(&render_pipeline);
                rpass.set_immediates(0, data);
                rpass.draw(0..1, 0..1);
                rpass.execute_bundles([&bundle]);
            }
            ctx.queue.submit([encoder.finish()]);
        }
    }

    ctx.async_poll(PollType::wait_indefinitely()).await.unwrap();
}

/// Immediates set while a pipeline with a smaller immediate size is bound must reach a
/// later pipeline with a larger immediate size.
#[apply(gpu_test!)]
static LARGER_LAYOUT_AFTER_DRAW: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::IMMEDIATES)
            .limits(Limits {
                max_immediate_size: 8,
                ..Default::default()
            }),
    )
    .run_async(|ctx| async move {
        for use_render_bundle in [false, true] {
            larger_layout_after_draw_test(&ctx, use_render_bundle).await;
        }
    });

async fn larger_layout_after_draw_test(ctx: &TestingContext, use_render_bundle: bool) {
    let module = create_wgsl_module(ctx, LAYOUT_SIZE_SHADER);
    let view = create_target(ctx);
    let output = Output::new(ctx, ShaderStages::FRAGMENT, 8);
    let small_layout = create_pipeline_layout(ctx, &[], 4);
    let large_layout = create_pipeline_layout(ctx, &[Some(&output.bind_group_layout)], 8);
    let small_pipeline = create_point_pipeline(ctx, &module, &small_layout, "fs_empty");
    let large_pipeline = create_point_pipeline(ctx, &module, &large_layout, "fs_write");

    fn do_encoding<'a>(
        encoder: &mut dyn RenderEncoder<'a>,
        small_pipeline: &'a RenderPipeline,
        large_pipeline: &'a RenderPipeline,
        bind_group: &'a BindGroup,
    ) {
        encoder.set_pipeline(small_pipeline);
        encoder.set_immediates(0, bytemuck::bytes_of(&[1_u32, 2]));
        encoder.draw(0..1, 0..1);
        encoder.set_pipeline(large_pipeline);
        encoder.set_bind_group(0, Some(bind_group), &[]);
        encoder.draw(0..1, 0..1);
    }

    let mut encoder = create_command_encoder(ctx);
    {
        let mut rpass = begin_render_pass(&mut encoder, &view);
        if use_render_bundle {
            let mut bundle_encoder = create_bundle_encoder(ctx);
            do_encoding(
                &mut bundle_encoder,
                &small_pipeline,
                &large_pipeline,
                &output.bind_group,
            );
            let bundle = bundle_encoder.finish(&RenderBundleDescriptor::default());
            rpass.execute_bundles([&bundle]);
        } else {
            do_encoding(
                &mut rpass,
                &small_pipeline,
                &large_pipeline,
                &output.bind_group,
            );
        }
    }
    assert_eq!(output.read::<u32>(ctx, encoder).await, [1, 2]);
}

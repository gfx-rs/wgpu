use std::mem::size_of;
use std::num::NonZeroU64;

use wgpu::util::RenderEncoder;
use wgpu::*;
use wgpu_test::{gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

/// We want to test that partial updates to push constants work as expected.
///
/// As such, we dispatch two compute passes, one which writes the values
/// before a partial update, and one which writes the values after the partial update.
///
/// If the update code is working correctly, the values not written to by the second update
/// will remain unchanged.
#[gpu_test]
static PARTIAL_UPDATE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(wgpu::Features::PUSH_CONSTANTS)
            .limits(wgpu::Limits {
                max_push_constant_size: 32,
                ..Default::default()
            }),
    )
    .run_async(partial_update_test);

const SHADER: &str = r#"
    struct Pc {
        offset: u32,
        vector: vec4f,
    }

    var<push_constant> pc: Pc;

    @group(0) @binding(0)
    var<storage, read_write> output: array<vec4f>;

    @compute @workgroup_size(1)
    fn main() {
        output[pc.offset] = pc.vector;
    }
"#;

async fn partial_update_test(ctx: TestingContext) {
    let sm = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bind_group_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });

    let gpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gpu_buffer"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("cpu_buffer"),
        size: 32,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind_group"),
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: gpu_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline_layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..32,
            }],
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pipeline"),
            layout: Some(&pipeline_layout),
            module: &sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("encoder"),
        });

    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("compute_pass"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);

        // -- Dispatch 0 --

        // Dispatch number
        cpass.set_push_constants(0, bytemuck::bytes_of(&[0_u32]));
        // Update the whole vector.
        cpass.set_push_constants(16, bytemuck::bytes_of(&[1.0_f32, 2.0, 3.0, 4.0]));
        cpass.dispatch_workgroups(1, 1, 1);

        // -- Dispatch 1 --

        // Dispatch number
        cpass.set_push_constants(0, bytemuck::bytes_of(&[1_u32]));
        // Update just the y component of the vector.
        cpass.set_push_constants(20, bytemuck::bytes_of(&[5.0_f32]));
        cpass.dispatch_workgroups(1, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&gpu_buffer, 0, &cpu_buffer, 0, 32);
    ctx.queue.submit([encoder.finish()]);
    cpu_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    let data = cpu_buffer.slice(..).get_mapped_range();

    let floats: &[f32] = bytemuck::cast_slice(&data);

    // first 4 floats the initial value
    // second 4 floats the first update
    assert_eq!(floats, [1.0, 2.0, 3.0, 4.0, 1.0, 5.0, 3.0, 4.0]);
}
#[gpu_test]
static RENDER_PASS_TEST: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(
        TestParameters::default()
            .features(Features::PUSH_CONSTANTS | Features::VERTEX_WRITABLE_STORAGE)
            .limits(wgpu::Limits {
                max_push_constant_size: 64,
                ..Default::default()
            }),
    )
    .run_async(move |ctx| async move {
        for use_render_bundle in [false, true] {
            render_pass_test(&ctx, use_render_bundle).await;
        }
    });

// This shader simply  moves the values from vector_constants and push_constants into the
// result buffer.  It expects to be called 4 times (with vector_index in 0..4) with its
// topology being PointList, so that each vertex shader call leads to exactly one fragment
// call.
const SHADER2: &str = "
    const POSITION: vec4f = vec4f(0, 0, 0, 1);

    struct PushConstants {
        vertex_constants: vec4i,
        fragment_constants: vec4i,
    }

    var<push_constant> push_constants: PushConstants;

    @group(0) @binding(0) var<storage, read_write> result: array<i32>;

    struct VertexOutput {
        @builtin(position) position: vec4f,
        @location(0) index: u32,
    }

    @vertex fn vertex(
        @builtin(vertex_index) ix: u32,
    ) -> VertexOutput {
        result[ix] = push_constants.vertex_constants[ix];
        return VertexOutput(POSITION, ix);
    }

    @fragment fn fragment(
        @location(0) ix: u32,
     ) -> @location(0) vec4f {
        result[ix + 4u] = push_constants.fragment_constants[ix];
        return vec4f();
    }
";

async fn render_pass_test(ctx: &TestingContext, use_render_bundle: bool) {
    let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("output buffer"),
        size: 8 * size_of::<u32>() as BufferAddress,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let cpu_buffer = ctx.device.create_buffer(&BufferDescriptor {
        label: Some("cpu buffer"),
        size: output_buffer.size(),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // We need an output texture, even though we're not ever going to look at it.
    let output_texture = ctx.device.create_texture(&TextureDescriptor {
        size: Extent3d {
            width: 2,
            height: 2,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::RENDER_ATTACHMENT,
        label: Some("Output Texture"),
        view_formats: &[],
    });
    let output_texture_view = output_texture.create_view(&Default::default());

    let shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("Shader"),
        source: ShaderSource::Wgsl(SHADER2.into()),
    });

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::VERTEX_FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

    let render_pipeline_layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[PushConstantRange {
                stages: ShaderStages::VERTEX_FRAGMENT,
                range: 0..8 * size_of::<u32>() as u32,
            }],
            ..Default::default()
        });

    let pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: None,
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: None,
                targets: &[Some(output_texture.format().into())],
                compilation_options: Default::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::PointList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
            cache: None,
        });

    let render_pass_desc = RenderPassDescriptor {
        label: Some("Render Pass"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &output_texture_view,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::default()),
                store: StoreOp::Store,
            },
        })],
        ..Default::default()
    };

    let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
        label: Some("bind group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[BindGroupEntry {
            binding: 0,
            resource: output_buffer.as_entire_binding(),
        }],
    });

    let data: Vec<i32> = (0..8).map(|i| (i * i) - 1).collect();

    fn do_encoding<'a>(
        encoder: &mut dyn RenderEncoder<'a>,
        pipeline: &'a RenderPipeline,
        bind_group: &'a BindGroup,
        data: &'a Vec<i32>,
    ) {
        let data_as_u8: &[u8] = bytemuck::cast_slice(data.as_slice());
        encoder.set_pipeline(pipeline);
        encoder.set_push_constants(ShaderStages::VERTEX_FRAGMENT, 0, data_as_u8);
        encoder.set_bind_group(0, Some(bind_group), &[]);
        encoder.draw(0..4, 0..1);
    }

    let mut command_encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut render_pass = command_encoder.begin_render_pass(&render_pass_desc);
        if use_render_bundle {
            // Execute the commands in a render_bundle_encoder.
            let mut render_bundle_encoder =
                ctx.device
                    .create_render_bundle_encoder(&RenderBundleEncoderDescriptor {
                        color_formats: &[Some(output_texture.format())],
                        sample_count: 1,
                        ..RenderBundleEncoderDescriptor::default()
                    });
            do_encoding(&mut render_bundle_encoder, &pipeline, &bind_group, &data);
            let render_bundle = render_bundle_encoder.finish(&RenderBundleDescriptor::default());
            render_pass.execute_bundles([&render_bundle]);
        } else {
            // Execute the commands directly.
            do_encoding(&mut render_pass, &pipeline, &bind_group, &data);
        }
    }
    // Move the result to the cpu buffer, so that we can read them.
    command_encoder.copy_buffer_to_buffer(&output_buffer, 0, &cpu_buffer, 0, output_buffer.size());
    let command_buffer = command_encoder.finish();
    ctx.queue.submit([command_buffer]);
    cpu_buffer.slice(..).map_async(MapMode::Read, |_| ());
    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();
    let mapped_data = cpu_buffer.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice::<u8, i32>(&mapped_data).to_vec();
    drop(mapped_data);
    cpu_buffer.unmap();
    assert_eq!(&result, &data);
}

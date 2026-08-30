//! Render-pass coverage for the resource table: fragment-stage `getResource`
//! sampling in a draw, access from *both* the vertex and fragment stages, and a
//! depth-format texture view stored in a slot and sampled. The render pass
//! binds the table through [`RenderPassDescriptor::resource_table`].

use wgpu::*;
use wgpu_test::{
    apply, gpu_test, image::ReadbackBuffers, GpuTestConfiguration, GpuTestInitializer,
};

use super::common::{make_red_texture, read_u32s, table_params, texture_red};

pub fn all_tests(tests: &mut Vec<GpuTestInitializer>) {
    tests.extend([
        RESOURCE_TABLE_FRAGMENT_SAMPLING,
        RESOURCE_TABLE_VERTEX_AND_FRAGMENT,
        RESOURCE_TABLE_DEPTH_TEXTURE,
    ]);
}

/// A full-screen triangle whose fragment shader reads a table slot with
/// `getResource<texture_2d<f32>>` and `textureLoad`s it, writing the texel to a
/// 1x1 color target. Verifies the render-pass-descriptor table binding path and
/// fragment-stage table access.
#[apply(gpu_test!)]
static RESOURCE_TABLE_FRAGMENT_SAMPLING: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED: u8 = texture_red(3); // 40

        let (_texture, view) = make_red_texture(&ctx, RED);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view).expect("bind texture");

        let shader = r#"
enable resource_table;
@vertex
fn vs(@builtin(vertex_index) id: u32) -> @builtin(position) vec4<f32> {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4<f32>(p[id], 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> {
    let tex = getResource<texture_2d<f32>>(0u);
    return textureLoad(tex, vec2<i32>(0, 0), 0);
}
"#;
        let color = draw_fullscreen(&ctx, shader, &table);
        color.assert_buffer_contents(&ctx, &[RED, 0, 0, 255]).await;
    });

/// `getResource` accessed from **both** the vertex and fragment stages of one
/// pipeline. The vertex shader samples slot 0 and forwards the red channel as a
/// flat varying; the fragment shader samples slot 1. The output color's red
/// channel comes from the vertex-stage read and its green channel from the
/// fragment-stage read, so a correct result proves both stages reached the
/// table (whose Vulkan descriptor set is created with `ShaderStageFlags::ALL`).
#[apply(gpu_test!)]
static RESOURCE_TABLE_VERTEX_AND_FRAGMENT: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        const RED_V: u8 = texture_red(0); // 10, read in the vertex stage
        const RED_F: u8 = texture_red(4); // 50, read in the fragment stage

        let (_tv, view_v) = make_red_texture(&ctx, RED_V);
        let (_tf, view_f) = make_red_texture(&ctx, RED_F);
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &view_v).expect("bind vertex texture");
        table.update(1, &view_f).expect("bind fragment texture");

        let shader = r#"
enable resource_table;
struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) @interpolate(flat) v: f32,
}
@vertex
fn vs(@builtin(vertex_index) id: u32) -> VOut {
    var p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    let tex = getResource<texture_2d<f32>>(0u);
    let texel = textureLoad(tex, vec2<i32>(0, 0), 0);
    var out: VOut;
    out.pos = vec4<f32>(p[id], 0.0, 1.0);
    out.v = texel.r;
    return out;
}
@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let tex = getResource<texture_2d<f32>>(1u);
    let texel = textureLoad(tex, vec2<i32>(0, 0), 0);
    return vec4<f32>(in.v, texel.r, 0.0, 1.0);
}
"#;
        let color = draw_fullscreen(&ctx, shader, &table);
        color
            .assert_buffer_contents(&ctx, &[RED_V, RED_F, 0, 255])
            .await;
    });

/// A depth-format (`Depth32Float`) texture view stored in a table slot and
/// sampled. Depth textures are in M0 scope. A first submission clears the depth
/// texture to a known value via a depth-only render pass; a second submission
/// samples it with `getResource<texture_depth_2d>` + `textureLoad` in a compute
/// shader and reads back the decoded value.
#[apply(gpu_test!)]
static RESOURCE_TABLE_DEPTH_TEXTURE: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(table_params())
    .run_async(|ctx| async move {
        // Depth cleared to 0.25 -> round(0.25 * 255) = 64 (no tie-breaking).
        const DEPTH: f32 = 0.25;
        const EXPECTED: u32 = 64;

        let depth_texture = ctx.device.create_texture(&TextureDescriptor {
            label: Some("depth member"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth32Float,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let depth_view = depth_texture.create_view(&TextureViewDescriptor::default());

        // Submission 1: establish a known depth value via a clear-only pass.
        {
            let mut encoder = ctx
                .device
                .create_command_encoder(&CommandEncoderDescriptor { label: None });
            let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("depth clear"),
                color_attachments: &[],
                depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(Operations {
                        load: LoadOp::Clear(DEPTH),
                        store: StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
                resource_table: None,
            });
            drop(_pass);
            ctx.queue.submit(Some(encoder.finish()));
        }

        // Submission 2: sample the depth texture from the table.
        let table = ctx.device.create_resource_table(&ResourceTableDescriptor {
            label: None,
            size: 4,
        });
        table.update(0, &depth_view).expect("bind depth view");

        let shader = r#"
enable resource_table;
@group(0) @binding(0)
var<storage, read_write> output: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let tex = getResource<texture_depth_2d>(0u);
    let d = textureLoad(tex, vec2<i32>(0, 0), 0);
    output[0] = u32(round(d * 255.0));
}
"#;
        let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(shader.into()),
        });
        let output_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback_buffer = ctx.device.create_buffer(&BufferDescriptor {
            label: None,
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = ctx
            .device
            .create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: None,
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: output_buffer.as_entire_binding(),
            }],
        });
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
                uses_resource_table: true,
            });
        let pipeline = ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions::default(),
                cache: None,
            });

        let mut encoder = ctx
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_resource_table(Some(&table));
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_buffer, 0, 4);
        ctx.queue.submit(Some(encoder.finish()));

        let got = read_u32s(&ctx, &readback_buffer).await;
        assert_eq!(
            got,
            vec![EXPECTED],
            "sampled depth {DEPTH} should decode to {EXPECTED}"
        );
    });

/// Create a 1x1 `Rgba8Unorm` render target with `RENDER_ATTACHMENT | COPY_SRC`.
fn render_texture(ctx: &wgpu_test::TestingContext) -> Texture {
    ctx.device.create_texture(&TextureDescriptor {
        label: Some("resource-table render target"),
        size: Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Draw a full-screen triangle with the given `shader` (entry points `vs`/`fs`)
/// and `table` bound through the render-pass descriptor, into a fresh 1x1
/// target. Returns the [`ReadbackBuffers`] for the target (already submitted).
fn draw_fullscreen(
    ctx: &wgpu_test::TestingContext,
    shader: &str,
    table: &ResourceTable,
) -> ReadbackBuffers {
    let module = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(shader.into()),
    });
    let layout = ctx
        .device
        .create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
            uses_resource_table: true,
        });
    let pipeline = ctx
        .device
        .create_render_pipeline(&RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(ColorTargetState {
                    format: TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

    let target = render_texture(ctx);
    let target_view = target.create_view(&TextureViewDescriptor::default());

    let mut encoder = ctx
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
            resource_table: Some(table),
        });
        pass.set_pipeline(&pipeline);
        pass.draw(0..3, 0..1);
    }
    let readback = ReadbackBuffers::new(&ctx.device, &target);
    readback.copy_from(&ctx.device, &mut encoder, &target);
    ctx.queue.submit(Some(encoder.finish()));

    readback
}

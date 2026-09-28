use core::num::NonZeroU32;

use wgpu::*;
use wgpu_test::fail;

fn check_visibility_requires_mesh_shader_feature(visibility: ShaderStages) {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor::default());

    fail(
        &device,
        || {
            device.create_bind_group_layout(&BindGroupLayoutDescriptor {
                label: Some("mesh shader visibility"),
                entries: &[BindGroupLayoutEntry {
                    binding: 0,
                    visibility,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            })
        },
        Some("EXPERIMENTAL_MESH_SHADER"),
    );
}

#[test]
fn task_visibility_requires_mesh_shader_feature() {
    check_visibility_requires_mesh_shader_feature(ShaderStages::TASK);
}

#[test]
fn mesh_visibility_requires_mesh_shader_feature() {
    check_visibility_requires_mesh_shader_feature(ShaderStages::MESH);
}

const MESH_SHADER: &str = "
enable wgpu_mesh_shader;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}
struct PrimitiveOutput {
    @builtin(triangle_indices) indices: vec3<u32>,
}
struct MeshOutput {
    @builtin(vertices) vertices: array<VertexOutput, 3>,
    @builtin(primitives) primitives: array<PrimitiveOutput, 1>,
    @builtin(vertex_count) vertex_count: u32,
    @builtin(primitive_count) primitive_count: u32,
}

var<workgroup> mesh_output: MeshOutput;

@mesh(mesh_output)
@workgroup_size(1)
fn ms_main() {
    mesh_output.vertex_count = 3;
    mesh_output.primitive_count = 1;
    mesh_output.primitives[0].indices = vec3<u32>(0, 1, 2);
}
";

/// Records a mesh draw in a render pass that renders to the views in `multiview_mask`, on a
/// device with the given `max_mesh_multiview_view_count`.
fn mesh_draw_in_multiview_pass(
    mesh_shader_multiview: bool,
    max_mesh_multiview_view_count: u32,
    multiview_mask: u32,
    expected_error: Option<&str>,
) {
    let mut features =
        Features::EXPERIMENTAL_MESH_SHADER | Features::MULTIVIEW | Features::SELECTIVE_MULTIVIEW;
    if mesh_shader_multiview {
        features |= Features::EXPERIMENTAL_MESH_SHADER_MULTIVIEW;
    }
    let (device, _queue) = Device::noop(&DeviceDescriptor {
        required_features: features,
        required_limits: Limits {
            max_mesh_multiview_view_count,
            max_multiview_view_count: 4,
            ..Limits::defaults().using_recommended_minimum_mesh_shader_values()
        },
        // SAFETY: the noop backend does not execute anything.
        experimental_features: unsafe { ExperimentalFeatures::enabled() },
        ..Default::default()
    });

    let multiview_mask = NonZeroU32::new(multiview_mask);
    let layers = 32 - multiview_mask.unwrap().leading_zeros();

    let shader = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(MESH_SHADER.into()),
    });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[],
        immediate_size: 0,
    });
    let pipeline = device.create_mesh_pipeline(&MeshPipelineDescriptor {
        label: None,
        layout: Some(&layout),
        task: None,
        mesh: MeshState {
            module: &shader,
            entry_point: Some("ms_main"),
            compilation_options: Default::default(),
        },
        fragment: None,
        primitive: PrimitiveState::default(),
        depth_stencil: Some(DepthStencilState {
            format: TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(CompareFunction::Always),
            stencil: StencilState::default(),
            bias: DepthBiasState::default(),
        }),
        multisample: MultisampleState::default(),
        multiview: multiview_mask,
        cache: None,
    });

    let depth = device.create_texture(&TextureDescriptor {
        label: None,
        size: Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: layers,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..Default::default()
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());
    {
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask,
        });
        pass.set_pipeline(&pipeline);
        pass.draw_mesh_tasks(1, 1, 1);
    }

    match expected_error {
        Some(msg) => {
            fail(&device, || encoder.finish(), Some(msg));
        }
        None => {
            encoder.finish();
        }
    }
}

#[test]
fn mesh_draw_multiview_requires_mesh_shader_multiview_feature() {
    mesh_draw_in_multiview_pass(false, 4, 0b11, Some("EXPERIMENTAL_MESH_SHADER_MULTIVIEW"));
}

#[test]
fn mesh_draw_multiview_highest_view_below_limit() {
    // Views 0 and 1 with a limit of 2 views.
    mesh_draw_in_multiview_pass(true, 2, 0b11, None);
}

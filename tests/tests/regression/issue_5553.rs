use wgpu_test::{gpu_test, GpuTestConfiguration};

use wgpu::*;

/// Previously, for every user-defined vertex output a fragment shader had to have a corresponding
/// user-defined input. This would generate `StageError::InputNotConsumed`.
///
/// This requirement was removed from the WebGPU spec. Now, when generating hlsl, wgpu will
/// automatically remove any user-defined outputs from the vertex shader that are not present in
/// the fragment inputs. This is necessary for generating correct hlsl:
/// https://github.com/gfx-rs/wgpu/issues/5553
#[gpu_test]
static ALLOW_INPUT_NOT_CONSUMED: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let module = ctx
            .device
            .create_shader_module(include_wgsl!("issue_5553.wgsl"));

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&PipelineLayoutDescriptor {
                label: Some("Pipeline Layout"),
                bind_group_layouts: &[],
                push_constant_ranges: &[],
            });

        let _ = ctx
            .device
            .create_render_pipeline(&RenderPipelineDescriptor {
                label: Some("Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: PrimitiveState::default(),
                depth_stencil: None,
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: ColorWrites::all(),
                    })],
                }),
                multiview: None,
                cache: None,
            });
    });

use wgpu::*;
use wgpu_test::{fail, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

const VERTEX_SHADER: &str = r#"
@vertex
fn vs_main() -> @builtin(position) vec4f {
    return vec4f(1.0);
}
"#;

const FRAGMENT_SHADER_WITHOUT_DUAL_SOURCE_BLENDING: &str = r#"
@fragment 
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0);
}
"#;

const FRAGMENT_SHADER_WITH_DUAL_SOURCE_BLENDING: &str = r#"
enable dual_source_blending;
struct FragmentOutput {
    @location(0) @blend_src(0) output0_: vec4<f32>,
    @location(0) @blend_src(1) output1_: vec4<f32>,
}

@fragment 
fn fs_main() -> FragmentOutput {
    return FragmentOutput(vec4<f32>(0.4f, 0.3f, 0.2f, 0.1f), vec4<f32>(0.9f, 0.8f, 0.7f, 0.6f));
}
"#;

fn blend_state_with_dual_source_blending() -> BlendState {
    wgpu::BlendState {
        // "random" blend factors using a second blend source.
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Src1,
            dst_factor: wgpu::BlendFactor::Src1Alpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Src1Alpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrc1Alpha,
            operation: wgpu::BlendOperation::Subtract,
        },
    }
}

#[gpu_test]
static DUAL_SOURCE_BLENDING_FEATURE_DISABLED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default())
    .run_async(dual_source_blending_disabled);

async fn dual_source_blending_disabled(ctx: TestingContext) {
    let vertex_shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vertex_shader"),
        source: ShaderSource::Wgsl(VERTEX_SHADER.into()),
    });
    let fragment_shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("fragment_shader"),
        source: ShaderSource::Wgsl(FRAGMENT_SHADER_WITHOUT_DUAL_SOURCE_BLENDING.into()),
    });

    // Can't create a render pipeline using blend modes that require dual source blending.
    fail(
        &ctx.device,
        || {
            let _ = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("render_pipeline"),
                    layout: None,
                    fragment: Some(wgpu::FragmentState {
                        module: &fragment_shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(blend_state_with_dual_source_blending()),
                            write_mask: wgpu::ColorWrites::all(),
                        })],
                        compilation_options: Default::default(),
                    }),
                    vertex: wgpu::VertexState {
                        module: &vertex_shader,
                        entry_point: None,
                        buffers: &[],
                        compilation_options: Default::default(),
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });
        },
        Some("Features Features { features_wgpu: FeaturesWGPU(0x0), features_webgpu: FeaturesWebGPU(DUAL_SOURCE_BLENDING) } are required but not enabled on the device"),
    );

    // Can't create a shader using dual source blending.
    fail(
        &ctx.device,
        || {
            let _ = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: Some("shader"),
                source: ShaderSource::Wgsl(FRAGMENT_SHADER_WITH_DUAL_SOURCE_BLENDING.into()),
            });
        },
        Some("Capability Capabilities(DUAL_SOURCE_BLENDING) is not supported"),
    );
}

#[gpu_test]
static DUAL_SOURCE_BLENDING_FEATURE_ENABLED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().features(wgpu::Features::DUAL_SOURCE_BLENDING))
    .run_async(dual_source_blending_enabled);

async fn dual_source_blending_enabled(ctx: TestingContext) {
    let vertex_shader = ctx.device.create_shader_module(ShaderModuleDescriptor {
        label: Some("vertex_shader"),
        source: ShaderSource::Wgsl(VERTEX_SHADER.into()),
    });
    let fragment_shader_without_dual_source_blending =
        ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fragment_shader"),
            source: ShaderSource::Wgsl(FRAGMENT_SHADER_WITHOUT_DUAL_SOURCE_BLENDING.into()),
        });
    let fragment_shader_with_dual_source_blending =
        ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("fragment_shader"),
            source: ShaderSource::Wgsl(FRAGMENT_SHADER_WITH_DUAL_SOURCE_BLENDING.into()),
        });

    let render_pipeline_descriptor_template = wgpu::RenderPipelineDescriptor {
        label: Some("render_pipeline"),
        layout: None,
        fragment: Some(wgpu::FragmentState {
            module: &fragment_shader_without_dual_source_blending,
            entry_point: Some("fs_main"),
            targets: &[],
            compilation_options: Default::default(),
        }),
        vertex: wgpu::VertexState {
            module: &vertex_shader,
            entry_point: None,
            buffers: &[],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    };

    // Happy path:
    // blend operator dual source: yes
    // shader handling dual source: yes
    let _ = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            fragment: Some(wgpu::FragmentState {
                module: &fragment_shader_with_dual_source_blending,
                entry_point: None,
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: Some(blend_state_with_dual_source_blending()),
                    write_mask: wgpu::ColorWrites::all(),
                })],
                compilation_options: Default::default(),
            }),
            ..render_pipeline_descriptor_template.clone()
        });

    // Failure mode:
    // blend operator dual source: no
    // shader handling dual source: yes
    fail(
        &ctx.device,
        || {
            let _ = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    fragment: Some(wgpu::FragmentState {
                        module: &fragment_shader_with_dual_source_blending,
                        entry_point: None,
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: None,
                            write_mask: wgpu::ColorWrites::all(),
                        })],
                        compilation_options: Default::default(),
                    }),
                    ..render_pipeline_descriptor_template.clone()
                });
        },
        Some("Shader entry point expects the pipeline to make use of dual-source blending."),
    );

    // Failure mode:
    // blend operator dual source: yes
    // shader handling dual source: no
    fail(
        &ctx.device,
        || {
            let _ = ctx
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    fragment: Some(wgpu::FragmentState {
                        module: &fragment_shader_without_dual_source_blending,
                        entry_point: None,
                        targets: &[Some(wgpu::ColorTargetState {
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            blend: Some(blend_state_with_dual_source_blending()),
                            write_mask: wgpu::ColorWrites::all(),
                        })],
                        compilation_options: Default::default(),
                    }),
                    ..render_pipeline_descriptor_template.clone()
                });
        },
        Some("Pipeline expects the shader entry point to make use of dual-source blending."),
    );
}

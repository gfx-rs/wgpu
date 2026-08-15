//! Tests of [`wgpu::RenderPipeline`] and related.

use wgpu_test::{fail, valid};

#[test]
fn reject_fragment_shader_output_over_max_color_attachments() {
    let (device, _queue) = wgpu::Device::noop(&Default::default());

    // NOTE: Vertex shader is a boring quad. The fragment shader is the interesting part.
    let source = format!(
        "\
@vertex
fn vert(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4f {{
    var pos = array<vec2f, 3>(
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5)
    );
    return vec4f(pos[vertex_index], 0.0, 1.0);
}}

@fragment
fn frag() -> @location({}) vec4f {{
    return vec4(1.0, 0.0, 0.0, 1.0);
}}
",
        device.limits().max_color_attachments
    );

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let module = &module;

    fail(
        &device,
        || {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: None,
                label: None,
                vertex: wgpu::VertexState {
                    module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: None,
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: Default::default(),
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            })
        },
        Some(concat!(
            "Location[8] Float32x4 interpolated as Some(Perspective) ",
            "with sampling Some(Center)'s index exceeds the `max_color_attachments` limit (8)"
        )),
    );
}

/// A fragment shader may output an `f16` value to a color target whose sample
/// type is `float`, even when the format's components are 32-bit. The WebGPU
/// spec only requires the output's scalar *kind* (floating-point) to match the
/// format's sample type, not its bit width.
#[test]
fn accept_f16_fragment_output_to_f32_target() {
    let (device, _queue) = wgpu::Device::noop(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::SHADER_F16,
        ..Default::default()
    });

    // NOTE: Vertex shader is a boring quad. The fragment shader is the interesting part.
    let source = "\
enable f16;

@vertex
fn vert(@builtin(vertex_index) vertex_index : u32) -> @builtin(position) vec4f {
    var pos = array<vec2f, 3>(
        vec2(0.0, 0.5),
        vec2(-0.5, -0.5),
        vec2(0.5, -0.5)
    );
    return vec4f(pos[vertex_index], 0.0, 1.0);
}

@fragment
fn frag() -> @location(0) vec4h {
    return vec4h(1.0h, 0.0h, 0.0h, 1.0h);
}
";

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let module = &module;

    valid(&device, || {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            layout: None,
            label: None,
            vertex: wgpu::VertexState {
                module,
                entry_point: None,
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: Default::default(),
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        })
    });
}

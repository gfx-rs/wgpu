//! Tests of [`wgpu::RenderPipeline`] and related.

use wgpu_test::fail;

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
                multiview: None,
                cache: None,
            })
        },
        Some(concat!(
            "Location[8] Float32x4 interpolated as Some(Perspective) ",
            "with sampling Some(Center)'s index exceeds the `max_color_attachments` limit (8)"
        )),
    );
}

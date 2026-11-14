//! Validation tests for pipeline-overridable constants.

use wgpu_test::{fail, valid};

fn compute_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    constants: &[(&str, f64)],
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module,
        entry_point: None,
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    })
}

/// Workgroup size limits must be checked after override values are applied.
#[test]
fn workgroup_size_from_override() {
    let (device, _queue) = wgpu::Device::noop(&Default::default());

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "
            override size: u32 = 1;
            @compute @workgroup_size(size) fn main() {}
            "
            .into(),
        ),
    });

    valid(&device, || compute_pipeline(&device, &module, &[]));
    valid(&device, || {
        compute_pipeline(&device, &module, &[("size", 64.0)])
    });
    fail(
        &device,
        || compute_pipeline(&device, &module, &[("size", 1024.0)]),
        Some("must be less or equal to the per-dimension limit"),
    );
}

#[test]
fn unknown_override() {
    let (device, _queue) = wgpu::Device::noop(&Default::default());

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()),
    });

    fail(
        &device,
        || compute_pipeline(&device, &module, &[("nonexistent", 1.0)]),
        Some("pipeline constant error"),
    );
}

#[test]
fn missing_override_value_in_render_pipeline() {
    let (device, _queue) = wgpu::Device::noop(&Default::default());

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            "
            override depth: f32;
            @vertex fn vs() -> @builtin(position) vec4f { return vec4f(0.0, 0.0, depth, 1.0); }
            @fragment fn fs() -> @location(0) vec4f { return vec4f(1.0); }
            "
            .into(),
        ),
    });

    let render_pipeline = |constants: &[(&str, f64)]| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: None,
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    ..Default::default()
                },
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: None,
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview_mask: None,
            cache: None,
        })
    };

    valid(&device, || render_pipeline(&[("depth", 0.5)]));
    fail(
        &device,
        || render_pipeline(&[]),
        Some("pipeline constant error"),
    );
}

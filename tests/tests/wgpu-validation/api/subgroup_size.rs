//! Tests of the per-stage `SubgroupSize` field on `PipelineCompilationOptions`.

use wgpu::*;
use wgpu_test::fail;

const COMPUTE_SHADER: &str = "\
@compute @workgroup_size(1)
fn main() {}
";

const RENDER_SHADER: &str = "\
@vertex
fn vs_main() -> @builtin(position) vec4f {
    return vec4f(0.0, 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0);
}
";

#[test]
fn fixed_subgroup_size_must_be_power_of_two() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::SUBGROUP_SIZE_CONTROL,
        ..DeviceDescriptor::default()
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    fail(
        &device,
        || {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions {
                    subgroup_size: SubgroupSize::Fixed(7),
                    ..Default::default()
                },
                cache: None,
            })
        },
        Some("Subgroup size 7 is not a power of two"),
    );
}

#[test]
fn full_subgroups_rejected_on_render_pipeline() {
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor {
        required_features: Features::SUBGROUP_SIZE_CONTROL,
        ..DeviceDescriptor::default()
    });

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(RENDER_SHADER.into()),
    });

    fail(
        &device,
        || {
            device.create_render_pipeline(&RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    compilation_options: PipelineCompilationOptions {
                        subgroup_size: SubgroupSize::Full,
                        ..Default::default()
                    },
                    buffers: &[],
                },
                fragment: Some(FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(ColorTargetState {
                        format: TextureFormat::Rgba8Unorm,
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
        Some("`SubgroupSize::Full` is only valid on compute, task, and mesh stages"),
    );
}

#[test]
fn full_subgroups_reject_workgroup_size_below_subgroup_min_size() {
    // Use the noop backend with `subgroup_min_size = 8`. The compute shader
    // declares `@workgroup_size(4)`, which is below `subgroup_min_size`, so a
    // full subgroup cannot fit and pipeline creation must fail.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions {
                enable: true,
                subgroup_min_size: Some(8),
                subgroup_max_size: Some(64),
                ..Default::default()
            },
            ..Default::default()
        },
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("noop adapter");
    let (device, _queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        required_features: Features::SUBGROUP_SIZE_CONTROL,
        ..DeviceDescriptor::default()
    }))
    .expect("device");

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(
            "\
@compute @workgroup_size(4)
fn main() {}
"
            .into(),
        ),
    });

    fail(
        &device,
        || {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions {
                    subgroup_size: SubgroupSize::Full,
                    ..Default::default()
                },
                cache: None,
            })
        },
        Some("requires `@workgroup_size` x to be at least subgroup_min_size (8); got 4"),
    );
}

#[test]
fn fixed_subgroup_size_requires_subgroup_size_control_feature() {
    // Default features — `SUBGROUP_SIZE_CONTROL` is not requested.
    let (device, _queue) = wgpu::Device::noop(&DeviceDescriptor::default());

    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: None,
        source: ShaderSource::Wgsl(COMPUTE_SHADER.into()),
    });

    fail(
        &device,
        || {
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: PipelineCompilationOptions {
                    subgroup_size: SubgroupSize::Fixed(32),
                    ..Default::default()
                },
                cache: None,
            })
        },
        Some("SUBGROUP_SIZE_CONTROL"),
    );
}

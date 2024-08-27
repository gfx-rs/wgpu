use std::num::NonZeroU64;

use wgpu_test::{fail, gpu_test, GpuTestConfiguration, TestParameters, TestingContext};

const SHADER_SRC: &str = "
@group(0) @binding(0)
var<uniform> buffer : f32;

@compute @workgroup_size(1, 1, 1) fn no_resources() {}
@compute @workgroup_size(1, 1, 1) fn resources() {
    // Just need a static use.
    let _value = buffer;
}
";

const ENTRY: wgpu::BindGroupLayoutEntry = wgpu::BindGroupLayoutEntry {
    binding: 0,
    visibility: wgpu::ShaderStages::COMPUTE,
    ty: wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Uniform,
        has_dynamic_offset: false,
        // Should be Some(.unwrap()) but unwrap is not const.
        min_binding_size: NonZeroU64::new(4),
    },
    count: None,
};

#[gpu_test]
static BIND_GROUP_LAYOUT_DEDUPLICATION: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_async(bgl_dedupe);

async fn bgl_dedupe(ctx: TestingContext) {
    let entries = &[];

    let bgl_1a = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries,
        });

    let bgl_1b = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries,
        });

    let bg_1a = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_1a,
        entries: &[],
    });

    let bg_1b = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_1b,
        entries: &[],
    });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl_1b],
            push_constant_ranges: &[],
        });

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    let desc = wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("no_resources"),
        compilation_options: Default::default(),
        cache: None,
    };

    let pipeline = ctx.device.create_compute_pipeline(&desc);

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_bind_group(0, &bg_1b, &[]);
    pass.set_pipeline(&pipeline);
    pass.dispatch_workgroups(1, 1, 1);

    pass.set_bind_group(0, &bg_1a, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    drop(pass);

    ctx.queue.submit(Some(encoder.finish()));
}

#[gpu_test]
static BIND_GROUP_LAYOUT_DEDUPLICATION_WITH_DROPPED_USER_HANDLE: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_sync(bgl_dedupe_with_dropped_user_handle);

// https://github.com/gfx-rs/wgpu/issues/4824
fn bgl_dedupe_with_dropped_user_handle(ctx: TestingContext) {
    let bgl_1 = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ENTRY],
        });

    let pipeline_layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl_1],
            push_constant_ranges: &[],
        });

    // We drop bgl_1 here. As bgl_1 is still alive, referenced by the pipeline layout,
    // the deduplication should work as expected. Previously this did not work.
    drop(bgl_1);

    let bgl_2 = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ENTRY],
        });

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl_2,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("no_resources"),
            compilation_options: Default::default(),
            cache: None,
        });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_bind_group(0, &bg, &[]);
    pass.set_pipeline(&pipeline);
    pass.dispatch_workgroups(1, 1, 1);

    drop(pass);

    ctx.queue.submit(Some(encoder.finish()));
}

#[gpu_test]
static GET_DERIVED_BGL: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_sync(get_derived_bgl);

fn get_derived_bgl(ctx: TestingContext) {
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("resources"),
            compilation_options: Default::default(),
            cache: None,
        });

    // We create two bind groups, pulling the bind_group_layout from the pipeline each time.
    //
    // This ensures a derived BGLs are properly deduplicated despite multiple external
    // references.
    let bg1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let bg2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_pipeline(&pipeline);

    pass.set_bind_group(0, &bg1, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    pass.set_bind_group(0, &bg2, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    drop(pass);

    ctx.queue.submit(Some(encoder.finish()));
}

#[gpu_test]
static SEPARATE_PIPELINES_HAVE_INCOMPATIBLE_DERIVED_BGLS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_sync(separate_pipelines_have_incompatible_derived_bgls);

fn separate_pipelines_have_incompatible_derived_bgls(ctx: TestingContext) {
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    let desc = wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("resources"),
        compilation_options: Default::default(),
        cache: None,
    };
    // Create two pipelines, creating a BG from the second.
    let pipeline1 = ctx.device.create_compute_pipeline(&desc);
    let pipeline2 = ctx.device.create_compute_pipeline(&desc);

    let bg2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline2.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_pipeline(&pipeline1);

    // We use the wrong bind group for this pipeline here. This should fail.
    pass.set_bind_group(0, &bg2, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    fail(
        &ctx.device,
        || {
            drop(pass);
        },
        Some("label at index 0 is not compatible with the corresponding bindgrouplayout"),
    );
}

#[gpu_test]
static DERIVED_BGLS_INCOMPATIBLE_WITH_REGULAR_BGLS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_sync(derived_bgls_incompatible_with_regular_bgls);

fn derived_bgls_incompatible_with_regular_bgls(ctx: TestingContext) {
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(SHADER_SRC.into()),
        });

    // Create a pipeline.
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: Some("resources"),
            compilation_options: Default::default(),
            cache: None,
        });

    // Create a matching BGL
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[ENTRY],
        });

    // Create a bind group from the explicit BGL. This should be incompatible with the derived BGL used by the pipeline.
    let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });

    let mut encoder = ctx.device.create_command_encoder(&Default::default());

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_pipeline(&pipeline);

    pass.set_bind_group(0, &bg, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    fail(
        &ctx.device,
        || {
            drop(pass);
        },
        Some("label at index 0 is not compatible with the corresponding bindgrouplayout"),
    )
}

#[gpu_test]
static BIND_GROUP_LAYOUT_DEDUPLICATION_DERIVED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_sync(bgl_dedupe_derived);

fn bgl_dedupe_derived(ctx: TestingContext) {
    let src = "
        @group(0) @binding(0) var<uniform> u1: vec4f;
        @group(1) @binding(0) var<uniform> u2: vec4f;

        @compute @workgroup_size(1, 1, 1)
        fn main() {
            // Just need a static use.
            let _u1 = u1;
            let _u2 = u2;
        }
    ";
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });

    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &module,
            entry_point: None,
            compilation_options: Default::default(),
            cache: None,
        });

    let bind_group_layout_0 = pipeline.get_bind_group_layout(0);
    let bind_group_layout_1 = pipeline.get_bind_group_layout(1);

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM,
        mapped_at_creation: false,
    });

    let bind_group_0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout_1,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }],
    });
    let bind_group_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout_0,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buffer,
                offset: 0,
                size: None,
            }),
        }],
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline);
    pass.set_bind_group(0, &bind_group_0, &[]);
    pass.set_bind_group(1, &bind_group_1, &[]);
    pass.dispatch_workgroups(1, 1, 1);

    drop(pass);

    ctx.queue.submit(Some(encoder.finish()));
}

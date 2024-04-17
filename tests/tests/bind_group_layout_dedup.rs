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
    let entries_1 = &[];

    let entries_2 = &[ENTRY];

    // Block so we can force all resource to die.
    {
        let bgl_1a = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: entries_1,
            });

        let bgl_2 = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: entries_2,
            });

        let bgl_1b = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: entries_1,
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
            entry_point: "no_resources",
            compilation_options: Default::default(),
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

        // Abuse the fact that global_id is really just the bitpacked ids when targeting wgpu-core.
        if ctx.adapter_info.backend != wgt::Backend::BrowserWebGpu {
            let bgl_1a_idx = bgl_1a.global_id().inner() & 0xFFFF_FFFF;
            assert_eq!(bgl_1a_idx, 0);
            let bgl_2_idx = bgl_2.global_id().inner() & 0xFFFF_FFFF;
            assert_eq!(bgl_2_idx, 1);
            let bgl_1b_idx = bgl_1b.global_id().inner() & 0xFFFF_FFFF;
            assert_eq!(bgl_1b_idx, 2);
        }
    }

    ctx.async_poll(wgpu::Maintain::wait())
        .await
        .panic_on_timeout();

    if ctx.adapter_info.backend != wgt::Backend::BrowserWebGpu {
        // Indices are made reusable as soon as the handle is dropped so we keep them around
        // for the duration of the loop.
        let mut bgls = Vec::new();
        let mut indices = Vec::new();
        // Now all of the BGL ids should be dead, so we should get the same ids again.
        for _ in 0..=2 {
            let test_bgl = ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: entries_1,
                });

            let test_bgl_idx = test_bgl.global_id().inner() & 0xFFFF_FFFF;
            bgls.push(test_bgl);
            indices.push(test_bgl_idx);
        }
        // We don't guarantee that the IDs will appear in the same order. Sort them
        // and check that they all appear exactly once.
        indices.sort();
        for (i, index) in indices.iter().enumerate() {
            assert_eq!(*index, i as u64);
        }
    }
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
            entry_point: "no_resources",
            compilation_options: Default::default(),
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
static BIND_GROUP_LAYOUT_DEDUPLICATION_DERIVED: GpuTestConfiguration = GpuTestConfiguration::new()
    .parameters(TestParameters::default().test_features_limits())
    .run_sync(bgl_dedupe_derived);

fn bgl_dedupe_derived(ctx: TestingContext) {
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
            entry_point: "resources",
            compilation_options: Default::default(),
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
static SEPARATE_PROGRAMS_HAVE_INCOMPATIBLE_DERIVED_BGLS: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(TestParameters::default().test_features_limits())
        .run_sync(separate_programs_have_incompatible_derived_bgls);

fn separate_programs_have_incompatible_derived_bgls(ctx: TestingContext) {
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
        entry_point: "resources",
        compilation_options: Default::default(),
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

    fail(&ctx.device, || {
        drop(pass);
    });
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
            entry_point: "resources",
            compilation_options: Default::default(),
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

    fail(&ctx.device, || {
        drop(pass);
    })
}

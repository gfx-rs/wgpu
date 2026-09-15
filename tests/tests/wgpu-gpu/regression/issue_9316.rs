use std::num::NonZeroU64;

use wgpu::BufferUsages;
use wgpu_test::{
    apply, gpu_test, GpuTestConfiguration, GpuTestInitializer, TestParameters, TestingContext,
};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.push(DYNAMIC_OFFSET_BOUNDS_WITH_UNORDERED_ENTRIES);
}

/// Dynamic offsets are supplied in binding index order, but the entries of a
/// [`wgpu::BindGroupDescriptor`] may be listed in any order. Check that each
/// offset is bounds-checked against the binding it actually targets.
fn dynamic_offset_bounds_with_unordered_entries(ctx: TestingContext) {
    let align32 = ctx.device.limits().min_uniform_buffer_offset_alignment;
    let align64 = u64::from(align32);

    let make_buffer = |label, size| {
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: false,
        })
    };

    // Make some bindings that are definitely incompatible if checked in the wrong order.
    let tight = make_buffer("tight", align64);
    let slack = make_buffer("slack", align64 * 2);

    let make_bgle = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    };

    let bind_group_layout = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[make_bgle(0), make_bgle(1)],
        });

    let binding = |buffer| {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: Some(NonZeroU64::new(align64).unwrap()),
        })
    };

    // Start with binding 1, so specified entry order and binding index order are different.
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bg"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 1,
                resource: binding(&slack),
            },
            wgpu::BindGroupEntry {
                binding: 0,
                resource: binding(&tight),
            },
        ],
    });

    let encode = |offsets: &[wgpu::DynamicOffset]| {
        let mut encoder = ctx.device.create_command_encoder(&Default::default());
        {
            let mut cpass = encoder.begin_compute_pass(&Default::default());
            cpass.set_bind_group(0, &bind_group, offsets);
        }
        let _ = encoder.finish();
    };

    wgpu_test::valid(&ctx.device, || encode(&[0, align32]));

    let error_cases: &[(&[_], _)] = &[
        (
            &[align32, 0],
            concat!(
                "Dynamic binding offset index 0 with offset 256 ",
                "would overrun the buffer bound to BindGroup with 'bg' label 0 -> binding 0."
            ),
        ),
        (
            &[1, 0],
            concat!(
                "Dynamic binding index 0 (targeting BindGroup with 'bg' label 0, binding 0) ",
                "with value 1, does not respect ",
                "device's requested `min_uniform_buffer_offset_alignment` limit: 256"
            ),
        ),
        (
            &[0],
            concat!(
                "BindGroup with 'bg' label 0 expects 2 dynamic offsets. ",
                "However 1 dynamic offset were provided.",
            ),
        ),
        (
            &[0, align32, 0],
            concat!(
                "BindGroup with 'bg' label 0 expects 2 dynamic offsets. ",
                "However 3 dynamic offsets were provided.",
            ),
        ),
    ];
    for (case, msg) in error_cases {
        wgpu_test::fail(&ctx.device, || encode(case), Some(msg));
    }
}

#[apply(gpu_test!)]
static DYNAMIC_OFFSET_BOUNDS_WITH_UNORDERED_ENTRIES: GpuTestConfiguration =
    GpuTestConfiguration::new()
        .parameters(
            TestParameters::default()
                .limits(wgpu::Limits::downlevel_defaults())
                .enable_noop(),
        )
        .run_sync(dynamic_offset_bounds_with_unordered_entries);

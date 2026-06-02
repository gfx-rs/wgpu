//! Tests for `PassthroughShaderEntryPoint::subgroup_size` validation, which
//! runs at shader-module-creation time.
//!
//! All cases here use a passthrough SPIR-V module (with throw-away bytes) on
//! the noop backend. The subgroup-size validation runs before the noop backend
//! would reject the actual shader source, so the validation error surfaces
//! first.
//!
//! The stage-dependent check (`FullSubgroupsNotAllowed` for `Full` on a
//! vertex/fragment stage) is exercised on real backends in the gpu test suite,
//! since it requires a successfully-created passthrough module which noop does
//! not provide.

use std::borrow::Cow;

use wgpu::*;
use wgpu_test::fail;

fn noop_device_with_subgroup_range(
    required_features: Features,
    subgroup_min_size: u32,
    subgroup_max_size: u32,
) -> Device {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::NOOP,
        backend_options: wgpu::BackendOptions {
            noop: wgpu::NoopBackendOptions {
                enable: true,
                subgroup_min_size: Some(subgroup_min_size),
                subgroup_max_size: Some(subgroup_max_size),
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
        required_features: required_features | Features::PASSTHROUGH_SHADERS,
        ..DeviceDescriptor::default()
    }))
    .expect("device");
    device
}

fn dummy_spirv() -> Cow<'static, [u32]> {
    // Magic-number-only SPIR-V; the noop backend never executes the module,
    // and `validate_passthrough_subgroup_sizes` runs before the source is ever
    // consumed.
    Cow::Borrowed(&[0x07230203])
}

fn passthrough_descriptor(
    entry_points: &'static [PassthroughShaderEntryPoint<'static>],
    spirv: Option<Cow<'static, [u32]>>,
    wgsl: Option<Cow<'static, str>>,
) -> ShaderModuleDescriptorPassthrough<'static> {
    ShaderModuleDescriptorPassthrough {
        label: None,
        entry_points: Cow::Borrowed(entry_points),
        spirv,
        wgsl,
        ..Default::default()
    }
}

#[test]
fn fixed_subgroup_size_must_be_power_of_two() {
    let device = noop_device_with_subgroup_range(Features::SUBGROUP_SIZE_CONTROL, 4, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (32, 1, 1),
        subgroup_size: SubgroupSize::Fixed(7),
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                Some(dummy_spirv()),
                None,
            ))
        },
        Some("Subgroup size 7"),
    );
}

#[test]
fn fixed_subgroup_size_must_be_in_range() {
    let device = noop_device_with_subgroup_range(Features::SUBGROUP_SIZE_CONTROL, 8, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (32, 1, 1),
        subgroup_size: SubgroupSize::Fixed(4),
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                Some(dummy_spirv()),
                None,
            ))
        },
        Some("Subgroup size 4"),
    );
}

#[test]
fn full_subgroups_reject_workgroup_size_below_subgroup_min_size() {
    let device = noop_device_with_subgroup_range(Features::SUBGROUP_SIZE_CONTROL, 8, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (4, 1, 1),
        subgroup_size: SubgroupSize::Full,
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                Some(dummy_spirv()),
                None,
            ))
        },
        Some("requires `@workgroup_size` x to be at least subgroup_min_size (8); got 4"),
    );
}

#[test]
fn fixed_subgroup_size_rejects_workgroup_size_x_not_multiple() {
    let device = noop_device_with_subgroup_range(Features::SUBGROUP_SIZE_CONTROL, 8, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (8, 1, 1),
        subgroup_size: SubgroupSize::Fixed(16),
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                Some(dummy_spirv()),
                None,
            ))
        },
        Some("requires `@workgroup_size` x to be a multiple of 16; got 8"),
    );
}

#[test]
fn non_varying_requires_subgroup_size_control_feature() {
    let device = noop_device_with_subgroup_range(Features::empty(), 4, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (32, 1, 1),
        subgroup_size: SubgroupSize::Fixed(32),
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                Some(dummy_spirv()),
                None,
            ))
        },
        Some("SUBGROUP_SIZE_CONTROL"),
    );
}

#[test]
fn non_varying_requires_workgroup_size() {
    let device = noop_device_with_subgroup_range(Features::SUBGROUP_SIZE_CONTROL, 4, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (0, 0, 0),
        subgroup_size: SubgroupSize::Fixed(32),
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                Some(dummy_spirv()),
                None,
            ))
        },
        Some("requires `PassthroughShaderEntryPoint::workgroup_size`"),
    );
}

#[test]
fn non_varying_only_supported_for_spirv_source() {
    let device = noop_device_with_subgroup_range(Features::SUBGROUP_SIZE_CONTROL, 4, 64);

    static ENTRY: &[PassthroughShaderEntryPoint<'static>] = &[PassthroughShaderEntryPoint {
        name: Cow::Borrowed("main"),
        workgroup_size: (32, 1, 1),
        subgroup_size: SubgroupSize::Fixed(32),
    }];

    fail(
        &device,
        || unsafe {
            device.create_shader_module_passthrough(passthrough_descriptor(
                ENTRY,
                None,
                Some(Cow::Borrowed("@compute @workgroup_size(32) fn main() {}")),
            ))
        },
        Some("only supported for SPIR-V passthrough"),
    );
}

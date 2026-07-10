//! Validation tests for the wgpu-core resource table object (work item 0.6 of
//! the bindless feature), its encoder-state plumbing (work item 0.7), and the
//! queue machinery / host slot-update flows (work item 0.8).
//!
//! These run against the `noop` backend through the `wgpu-core` resource
//! methods directly (like [`limit_buckets`]), so that the validation surface is
//! exercised without going through the public `wgpu` wrappers.
//!
//! # noop coverage limits (work item 0.8)
//!
//! The noop backend completes submissions synchronously (its fence advances to
//! the submit index during `submit`, whose internal `maintain` then advances
//! the cached completed-submission index). So no submission is ever *in flight*
//! from the test's point of view, and the two in-flight-dependent behaviors —
//! `SlotInUse` gating while a submission that can reach the slot is pending, and
//! `destroy` deferral while a table is in flight — cannot be observed here. The
//! gate *logic* is covered by the `slot_reuse_gate` unit test in
//! `wgpu-core::resource_table`; the end-to-end in-flight behavior is exercised
//! by the GPU test suite (work item 0.12 / wave 6). What *is* covered here: the
//! full host update/insert/remove validation surface, and that submit runs the
//! marking + pass-start gap splice without error and leaves slots reusable once
//! completed.
//!
//! [`limit_buckets`]: crate::limit_buckets

use std::borrow::Cow;
use std::sync::Arc;

use wgpu_core as wgc;
use wgpu_types as wgt;

use wgc::binding_model::PipelineLayout;
use wgc::device::queue::Queue;
use wgc::device::Device;
use wgc::instance::{Adapter, Instance};
use wgc::pipeline::{
    ComputePipeline, CreateComputePipelineError, CreateShaderModuleError,
    ResourceTablePipelineError, ShaderModule,
};
use wgc::resource::{Texture, TextureView};
use wgc::resource_table::{
    CreateResourceTableError, ResourceTable, UpdateResourceTableError, MAX_RESOURCE_TABLE_SIZE,
};

fn create_noop_instance() -> Arc<Instance> {
    Instance::new(
        "resource_table_test",
        wgt::instance::InstanceDescriptor {
            backends: wgt::Backends::NOOP,
            backend_options: wgt::BackendOptions {
                noop: wgt::NoopBackendOptions {
                    enable: true,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..wgt::instance::InstanceDescriptor::new_without_display_handle()
        },
        None,
    )
}

fn noop_adapter(instance: &Arc<Instance>) -> Arc<Adapter> {
    instance
        .request_adapter(&wgt::RequestAdapterOptions::default(), wgt::Backends::NOOP)
        .expect("noop adapter should be available")
}

/// Request a device and its queue with the given required features enabled.
///
/// The queue is returned because a [`Device`] only holds a weak reference to
/// it, and dropping it makes the device unusable for command encoding.
///
/// Any resource-table feature is experimental, so the unsafe experimental gate
/// is passed whenever `features` is non-empty (all features requested by these
/// tests are experimental resource-table bits).
fn request_device_features(
    instance: &Arc<Instance>,
    features: wgt::Features,
) -> (Arc<Device>, Arc<Queue>) {
    let adapter = noop_adapter(instance);

    let experimental_features = if features.is_empty() {
        wgt::ExperimentalFeatures::disabled()
    } else {
        // SAFETY: This is a test; the noop backend has no actual unsafe
        // behavior behind the experimental gate.
        unsafe { wgt::ExperimentalFeatures::enabled() }
    };

    adapter
        .request_device(&wgt::DeviceDescriptor {
            required_features: features,
            experimental_features,
            ..Default::default()
        })
        .expect("device creation should succeed")
}

/// Request a device and its queue, optionally enabling the sampling resource
/// table feature.
fn request_device(instance: &Arc<Instance>, with_feature: bool) -> (Arc<Device>, Arc<Queue>) {
    let features = if with_feature {
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE
    } else {
        wgt::Features::empty()
    };
    request_device_features(instance, features)
}

/// Run `f` inside a validation error scope, returning its value along with the
/// error it reported to the device, if any.
///
/// `wgpu-core` reports creation and submission errors to the device error sink
/// instead of returning them, so tests observe them the same way an application
/// would.
fn with_error_scope<R>(
    device: &Arc<Device>,
    f: impl FnOnce() -> R,
) -> (R, Option<wgt::error::Error>) {
    device.push_error_scope(wgt::error::ErrorFilter::Validation);
    let value = f();
    let error = device
        .pop_error_scope()
        .expect("error scope stack should not be empty");
    (value, error)
}

fn descriptor(size: u32) -> wgc::resource_table::ResourceTableDescriptor<'static> {
    wgt::ResourceTableDescriptor { label: None, size }
}

/// Create a texture with the given usage and a default view of it. Returns both
/// the texture and the view.
fn create_texture_and_view(
    device: &Arc<Device>,
    usage: wgt::TextureUsages,
) -> (Arc<Texture>, Arc<TextureView>) {
    let (texture, error) = device.create_texture(&wgt::TextureDescriptor {
        label: None,
        size: wgt::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgt::TextureDimension::D2,
        format: wgt::TextureFormat::Rgba8Unorm,
        usage,
        view_formats: Vec::new(),
    });
    assert!(error.is_none(), "texture creation failed: {error:?}");

    let (view, error) = texture.create_view(&wgc::resource::TextureViewDescriptor {
        label: None,
        format: None,
        dimension: None,
        usage: None,
        range: wgt::ImageSubresourceRange::default(),
    });
    assert!(error.is_none(), "texture view creation failed: {error:?}");
    (texture, view)
}

/// A texture view with the given usage. The view keeps its parent texture
/// alive, so the caller need not hold on to it.
fn create_texture_view(device: &Arc<Device>, usage: wgt::TextureUsages) -> Arc<TextureView> {
    create_texture_and_view(device, usage).1
}

/// A sampled (`TEXTURE_BINDING`) texture view, valid for binding into a table.
fn create_sampled_view(device: &Arc<Device>) -> Arc<TextureView> {
    create_texture_view(device, wgt::TextureUsages::TEXTURE_BINDING)
}

/// Both resource-table features needed to create a pipeline whose shaders use
/// `getResource` in M0: the sampling feature (checked path plumbing) plus the
/// unchecked add-on (the only lowering that exists so far, D4).
fn sampling_and_unchecked() -> wgt::Features {
    wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE
        | wgt::Features::EXPERIMENTAL_RESOURCE_TABLE_UNCHECKED
}

/// A pipeline-layout descriptor with no bind groups and no immediates, whose
/// `uses_resource_table` flag is set as requested.
fn pipeline_layout_desc(
    uses_resource_table: bool,
) -> wgc::binding_model::PipelineLayoutDescriptor<'static> {
    wgc::binding_model::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: Cow::Owned(Vec::new()),
        immediate_size: 0,
        uses_resource_table,
    }
}

/// A minimal compute shader that reads a sampled texture from the resource
/// table via `getResource`, so its module reflects `uses_resource_table`. It
/// declares no ordinary bindings, so it is compatible with an empty layout.
const TABLE_COMPUTE_WGSL: &str = "\
enable resource_table;
var<private> sink: vec4<f32>;
@compute @workgroup_size(1)
fn main() {
    sink = textureLoad(getResource<texture_2d<f32>>(0u), vec2<i32>(0, 0), 0);
}
";

/// A trivial compute shader that does not touch the resource table.
const TRIVIAL_COMPUTE_WGSL: &str = "\
@compute @workgroup_size(1)
fn main() {}
";

/// A trivial vertex + fragment shader that does not touch the resource table.
/// Used to build a render pipeline whose *layout* declares a resource table (the
/// bundle-rejection flag comes from the layout, not the shader).
const TABLE_RENDER_WGSL: &str = "\
@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
";

fn create_shader(
    device: &Arc<Device>,
    wgsl: &str,
) -> (Arc<ShaderModule>, Option<CreateShaderModuleError>) {
    device.create_shader_module(
        &wgc::pipeline::ShaderModuleDescriptor {
            label: None,
            runtime_checks: wgt::ShaderRuntimeChecks::default(),
        },
        wgc::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(wgsl)),
    )
}

fn create_pipeline_layout(device: &Arc<Device>, uses_resource_table: bool) -> Arc<PipelineLayout> {
    device.create_pipeline_layout(&pipeline_layout_desc(uses_resource_table))
}

fn create_compute_pipeline(
    device: &Arc<Device>,
    layout: Option<Arc<PipelineLayout>>,
    module: Arc<ShaderModule>,
) -> Result<Arc<ComputePipeline>, CreateComputePipelineError> {
    device.create_compute_pipeline_or_error(wgc::pipeline::ComputePipelineDescriptor {
        label: None,
        layout,
        stage: wgc::pipeline::ProgrammableStageDescriptor {
            module,
            entry_point: Some(Cow::Borrowed("main")),
            constants: Default::default(),
            zero_initialize_workgroup_memory: false,
        },
        cache: None,
    })
}

/// Creating a resource table without the feature enabled fails with
/// `MissingFeatures`.
#[test]
fn create_without_feature_fails() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, false);

    let (_table, error) = device.create_resource_table(&descriptor(8));

    assert!(
        matches!(error, Some(CreateResourceTableError::MissingFeatures(_))),
        "expected MissingFeatures, got {error:?}"
    );
}

/// Creating a resource table with the feature enabled succeeds.
#[test]
fn create_with_feature_succeeds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(8));

    assert!(error.is_none(), "unexpected error: {error:?}");

    // Cleanup: destroy and drop should both be no-panic.
    table.destroy();
    drop(table);
}

/// A zero-slot resource table is rejected.
#[test]
fn create_zero_size_fails() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (_table, error) = device.create_resource_table(&descriptor(0));

    assert!(
        matches!(error, Some(CreateResourceTableError::ZeroSize)),
        "expected ZeroSize, got {error:?}"
    );
}

/// A resource table with more than `MAX_RESOURCE_TABLE_SIZE` slots is rejected,
/// while exactly `MAX_RESOURCE_TABLE_SIZE` is allowed.
#[test]
fn create_size_bounds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    // The maximum is allowed.
    let (max_table, error) = device.create_resource_table(&descriptor(MAX_RESOURCE_TABLE_SIZE));
    assert!(error.is_none(), "max size should be allowed, got {error:?}");
    drop(max_table);

    // One past the maximum is rejected.
    let (_table, error) = device.create_resource_table(&descriptor(MAX_RESOURCE_TABLE_SIZE + 1));
    assert!(
        matches!(
            error,
            Some(CreateResourceTableError::TooManySlots { max, .. })
                if max == MAX_RESOURCE_TABLE_SIZE
        ),
        "expected TooManySlots, got {error:?}"
    );
}

/// Destroying a resource table twice is a no-op (per spec), and dropping
/// afterwards does not panic.
#[test]
fn destroy_is_idempotent() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "unexpected error: {error:?}");

    table.destroy();
    // Destroying again must not panic.
    table.destroy();
    // Dropping after destroy must not panic.
    drop(table);
}

/// Dropping a resource table without destroying it first is fine (drop-based
/// cleanup path).
#[test]
fn drop_without_destroy() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "unexpected error: {error:?}");

    drop(table);
}

// ---------------------------------------------------------------------------
// Work item 0.7: pipeline-layout plumbing, shader/pipeline validation, and
// pass-scope `set_resource_table` with draw/dispatch-time checks.
// ---------------------------------------------------------------------------

/// Creating a pipeline layout with `uses_resource_table` set requires the
/// sampling resource table feature.
#[test]
fn pipeline_layout_flag_requires_feature() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(&instance, wgt::Features::empty());

    let (_layout, error) = with_error_scope(&device, || create_pipeline_layout(&device, true));

    let debug = format!("{error:?}");
    assert!(
        debug.contains("MissingFeatures"),
        "expected MissingFeatures, got {debug}"
    );
}

/// With the feature enabled, a `uses_resource_table` pipeline layout is created
/// successfully.
#[test]
fn pipeline_layout_flag_with_feature_succeeds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(
        &instance,
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
    );

    let (_layout, error) = with_error_scope(&device, || create_pipeline_layout(&device, true));

    assert!(error.is_none(), "unexpected error: {error:?}");
}

/// A shader that uses `getResource` fails to validate without the sampling
/// feature enabled (the `RESOURCE_TABLE` naga capability is gated on it).
#[test]
fn table_shader_requires_feature() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(&instance, wgt::Features::empty());

    let (_module, error) = create_shader(&device, TABLE_COMPUTE_WGSL);

    assert!(
        error.is_some(),
        "expected a shader validation error without the feature"
    );
}

/// A pipeline whose shader uses `getResource` must be created with a layout that
/// declares `uses_resource_table`; otherwise pipeline creation fails.
#[test]
fn table_shader_requires_table_layout() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(&instance, sampling_and_unchecked());

    let (module, error) = create_shader(&device, TABLE_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    // A layout that does *not* declare a resource table.
    let layout = create_pipeline_layout(&device, false);

    let error = create_compute_pipeline(&device, Some(layout), module).err();

    assert!(
        matches!(
            error,
            Some(CreateComputePipelineError::ResourceTable(
                ResourceTablePipelineError::LayoutMissingResourceTable
            ))
        ),
        "expected LayoutMissingResourceTable, got {error:?}"
    );
}

/// In M0 a pipeline whose shader uses `getResource` also requires the unchecked
/// feature (only the unchecked lowering exists so far, D4).
#[test]
fn table_shader_requires_unchecked_feature() {
    let instance = create_noop_instance();
    // Sampling only: enough to create the layout and validate the shader, but
    // not to create the (unchecked-only) pipeline.
    let (device, _queue) = request_device_features(
        &instance,
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
    );

    let (module, error) = create_shader(&device, TABLE_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let layout = create_pipeline_layout(&device, true);

    let error = create_compute_pipeline(&device, Some(layout), module).err();

    assert!(
        matches!(
            error,
            Some(CreateComputePipelineError::ResourceTable(
                ResourceTablePipelineError::MissingFeatures(_)
            ))
        ),
        "expected MissingFeatures, got {error:?}"
    );
}

/// A `getResource` shader with a matching `uses_resource_table` layout and both
/// features enabled creates a pipeline successfully (positive control).
#[test]
fn table_shader_with_table_layout_succeeds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(&instance, sampling_and_unchecked());

    let (module, error) = create_shader(&device, TABLE_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let layout = create_pipeline_layout(&device, true);

    let error = create_compute_pipeline(&device, Some(layout), module).err();

    assert!(error.is_none(), "expected success, got {error:?}");
}

/// Dispatching with a pipeline whose layout declares a resource table, without
/// binding one via `set_resource_table`, is a validation error (surfaced at
/// `CommandEncoder::finish`). The shader itself need not use `getResource`: the
/// check is on the layout flag, because the table's descriptor set is reserved
/// at the highest set index regardless.
#[test]
fn dispatch_without_bound_table_fails() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(
        &instance,
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
    );

    let (module, error) = create_shader(&device, TRIVIAL_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let layout = create_pipeline_layout(&device, true);

    let pipeline = create_compute_pipeline(&device, Some(layout), module)
        .expect("pipeline creation should succeed");

    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(1, 1, 1);
    pass.end();

    let (_cb, error) = with_error_scope(&device, || {
        encoder.finish(&wgt::CommandBufferDescriptor { label: None })
    });

    let debug = format!("{error:?}");
    assert!(
        debug.contains("ComputePass"),
        "expected a compute-pass error, got {debug}"
    );
    assert!(
        debug.contains("MissingResourceTable"),
        "expected MissingResourceTable, got {debug}"
    );
}

/// Binding a resource table before dispatch satisfies the layout requirement
/// and the command buffer finishes without error (positive control; also
/// exercises the hal `set_resource_table` emission path on the noop backend).
#[test]
fn dispatch_with_bound_table_succeeds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(
        &instance,
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
    );

    let (module, error) = create_shader(&device, TRIVIAL_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let layout = create_pipeline_layout(&device, true);

    let pipeline = create_compute_pipeline(&device, Some(layout), module)
        .expect("pipeline creation should succeed");

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_resource_table(Some(table));
    pass.set_pipeline(pipeline);
    pass.dispatch_workgroups(1, 1, 1);
    pass.end();

    let (_cb, error) = with_error_scope(&device, || {
        encoder.finish(&wgt::CommandBufferDescriptor { label: None })
    });

    assert!(error.is_none(), "expected finish to succeed, got {error:?}");
}

/// Binding a resource table created on a different device to a pass is a
/// validation error (surfaced at `CommandEncoder::finish`).
#[test]
fn resource_table_wrong_device_fails() {
    let instance = create_noop_instance();
    let (device_a, _queue_a) = request_device_features(
        &instance,
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
    );
    let (device_b, _queue_b) = request_device_features(
        &instance,
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
    );

    // Table on device A.
    let (table, error) = device_a.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    // Encoder / pass on device B.
    let encoder = device_b.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_resource_table(Some(table));
    pass.end();

    let (_cb, error) = with_error_scope(&device_b, || {
        encoder.finish(&wgt::CommandBufferDescriptor { label: None })
    });

    let debug = format!("{error:?}");
    assert!(
        debug.contains("ComputePass"),
        "expected a compute-pass error, got {debug}"
    );
    assert!(
        debug.contains("DeviceMismatch"),
        "expected DeviceMismatch, got {debug}"
    );
}

// ---------------------------------------------------------------------------
// Work item 0.8: host slot-update flows (`update`/`insert_binding`/
// `remove_binding`), queue-submit marking + gap splice, and render-bundle
// rejection.
// ---------------------------------------------------------------------------

/// Binding a sampled texture view into an in-bounds slot succeeds.
#[test]
fn update_slot_happy_path() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device);

    table.update_slot(0, &view).expect("update should succeed");
    // Overwriting a still-available slot is fine.
    table.update_slot(3, &view).expect("update should succeed");
}

/// Updating a slot outside the table's range fails with `SlotOutOfBounds`.
#[test]
fn update_slot_out_of_bounds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device);

    let result = table.update_slot(4, &view);
    assert!(
        matches!(
            result,
            Err(UpdateResourceTableError::SlotOutOfBounds { slot: 4, size: 4 })
        ),
        "expected SlotOutOfBounds, got {result:?}"
    );
}

/// A texture view without `TEXTURE_BINDING` usage (e.g. a render-attachment-only
/// texture) cannot be bound into a table in M0 (sampled/depth only).
#[test]
fn update_slot_wrong_texture_type() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    // Renderable but not sampleable: has a view but no `TEXTURE_BINDING`.
    let view = create_texture_view(
        &device,
        wgt::TextureUsages::RENDER_ATTACHMENT | wgt::TextureUsages::COPY_DST,
    );

    let result = table.update_slot(0, &view);
    assert!(
        matches!(
            result,
            Err(UpdateResourceTableError::MissingTextureUsage(_))
        ),
        "expected MissingTextureUsage, got {result:?}"
    );
}

/// Binding a texture view from a different device fails with a device mismatch.
#[test]
fn update_slot_wrong_device() {
    let instance = create_noop_instance();
    let (device_a, _queue_a) = request_device(&instance, true);
    let (device_b, _queue_b) = request_device(&instance, true);

    let (table, error) = device_a.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device_b);

    let result = table.update_slot(0, &view);
    assert!(
        matches!(result, Err(UpdateResourceTableError::Device(_))),
        "expected a device error, got {result:?}"
    );
}

/// Updating a slot of a destroyed table fails with `DestroyedResource`.
#[test]
fn update_destroyed_table() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device);

    table.destroy();

    let result = table.update_slot(0, &view);
    assert!(
        matches!(result, Err(UpdateResourceTableError::DestroyedResource(_))),
        "expected DestroyedResource, got {result:?}"
    );
}

/// Binding a view whose parent texture has been destroyed fails with
/// `DestroyedResource`.
#[test]
fn update_with_destroyed_texture() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let (texture, view) = create_texture_and_view(&device, wgt::TextureUsages::TEXTURE_BINDING);

    texture.destroy();

    let result = table.update_slot(0, &view);
    assert!(
        matches!(result, Err(UpdateResourceTableError::DestroyedResource(_))),
        "expected DestroyedResource, got {result:?}"
    );
}

/// `insert_binding` assigns the lowest-available slot (D8), and reports
/// `NoAvailableSlot` once the table is full.
#[test]
fn insert_binding_assigns_lowest_available() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(2));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device);

    let slot0 = table
        .insert_binding(&view)
        .expect("first insert should succeed");
    assert_eq!(slot0, 0);

    let slot1 = table
        .insert_binding(&view)
        .expect("second insert should succeed");
    assert_eq!(slot1, 1);

    // Table is now full.
    let result = table.insert_binding(&view);
    assert!(
        matches!(result, Err(UpdateResourceTableError::NoAvailableSlot)),
        "expected NoAvailableSlot, got {result:?}"
    );
}

/// Removing a binding frees its slot so the next `insert_binding` reuses it.
#[test]
fn remove_binding_frees_slot() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(2));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device);

    assert_eq!(table.insert_binding(&view).unwrap(), 0);
    assert_eq!(table.insert_binding(&view).unwrap(), 1);

    table.remove_binding(0).expect("remove should succeed");

    // The freed slot is the lowest available again.
    assert_eq!(table.insert_binding(&view).unwrap(), 0);
}

/// Removing an out-of-bounds slot is a validation error.
#[test]
fn remove_binding_out_of_bounds() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(2));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let result = table.remove_binding(2);
    assert!(
        matches!(
            result,
            Err(UpdateResourceTableError::SlotOutOfBounds { slot: 2, size: 2 })
        ),
        "expected SlotOutOfBounds, got {result:?}"
    );
}

/// End-to-end: bind a table (with a bound texture) in a compute pass, submit,
/// and confirm the submit — which runs the slot marking and the pass-start gap
/// splice (work item 0.8) — succeeds, and that the table's slots are reusable
/// once the (synchronously completed) submission is done.
///
/// On the noop backend the submission completes during `submit`, so this checks
/// the "available after completion" direction of the reuse gate and that the
/// marking + splice machinery runs cleanly. The "gated while in flight"
/// direction is not observable here (see the module docs).
#[test]
fn submit_marks_and_splices_then_slot_reusable() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view = create_sampled_view(&device);
    table.update_slot(0, &view).expect("update should succeed");

    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    pass.set_resource_table(Some(Arc::clone(&table)));
    pass.end();

    let cb = encoder.finish(&wgt::CommandBufferDescriptor { label: None });

    queue.submit(&[cb]);

    // The noop submission has completed; the slot is reusable.
    device
        .poll(wgt::PollType::Poll)
        .expect("poll should succeed");
    table
        .update_slot(0, &view)
        .expect("slot should be reusable after the submission completes");
}

/// A compute pass that binds two *distinct* tables (rebinding mid-pass) must
/// record a pass-start gap for **each** of them, not just the final binding
/// (finding M1). Both tables hold a bound texture, so both gaps carry real
/// barriers, exercising the multiple-gaps-at-one-insertion-point splice path.
///
/// The layout-barrier correctness itself is not observable on the noop backend
/// (its barriers are no-ops); what is checked here is that the marking + splice
/// runs cleanly for every bound table and that both tables' slots are reusable
/// once the (synchronously completed) submission is done. The barrier ordering
/// is exercised end-to-end by the GPU suite.
#[test]
fn compute_pass_rebound_tables_each_marked_and_spliced() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table_a, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table A creation failed: {error:?}");
    let (table_b, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table B creation failed: {error:?}");

    // Distinct member textures so each table's gap has a barrier to compute.
    let view_a = create_sampled_view(&device);
    let view_b = create_sampled_view(&device);
    table_a
        .update_slot(0, &view_a)
        .expect("update A should succeed");
    table_b
        .update_slot(0, &view_b)
        .expect("update B should succeed");

    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });

    // Bind A, then rebind B within the same pass. Before finding M1 only B (the
    // final binding) would have had a gap recorded.
    pass.set_resource_table(Some(Arc::clone(&table_a)));
    pass.set_resource_table(Some(Arc::clone(&table_b)));
    pass.end();

    let cb = encoder.finish(&wgt::CommandBufferDescriptor { label: None });

    queue.submit(&[cb]);
    device
        .poll(wgt::PollType::Poll)
        .expect("poll should succeed");

    // Both tables were marked and their gaps spliced; both slots are reusable
    // once the submission completes.
    table_a
        .update_slot(0, &view_a)
        .expect("table A slot should be reusable after completion");
    table_b
        .update_slot(0, &view_b)
        .expect("table B slot should be reusable after completion");
}

/// The same member texture bound in tables sampled by two separate passes of one
/// command buffer must have its pass-start transitions computed in execution
/// order (finding C1): the earlier pass transitions it to `RESOURCE`, the later
/// pass then observes it already there. This drives the ascending-compute /
/// descending-splice path with two gaps at *different* insertion points that
/// share a texture.
///
/// On the noop backend the barrier ordering is not observable (no-op barriers),
/// so this asserts only that the two-pass splice runs cleanly and the slots stay
/// reusable; the ordering itself is verified on real GPUs by the smoke suite.
#[test]
fn table_member_texture_shared_across_passes_splices_cleanly() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table1, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table 1 creation failed: {error:?}");
    let (table2, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table 2 creation failed: {error:?}");

    // One shared texture/view bound into both tables.
    let view = create_sampled_view(&device);
    table1
        .update_slot(0, &view)
        .expect("update table 1 should succeed");
    table2
        .update_slot(0, &view)
        .expect("update table 2 should succeed");

    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    // Pass 1 binds table1; pass 2 binds table2 — both reference the same texture.
    for table in [&table1, &table2] {
        let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_resource_table(Some(Arc::clone(table)));
        pass.end();
    }

    let cb = encoder.finish(&wgt::CommandBufferDescriptor { label: None });

    queue.submit(&[cb]);
    device
        .poll(wgt::PollType::Poll)
        .expect("poll should succeed");

    table1
        .update_slot(0, &view)
        .expect("table 1 slot should be reusable after completion");
    table2
        .update_slot(0, &view)
        .expect("table 2 slot should be reusable after completion");
}

/// A render bundle that records a pipeline whose layout declares a resource
/// table is rejected at bundle creation in M0 (2026-07-09 user decision).
#[test]
fn table_using_render_bundle_rejected() {
    let instance = create_noop_instance();
    let (device, _queue) = request_device_features(&instance, sampling_and_unchecked());

    // A trivial render pipeline whose *layout* uses a resource table (its shader
    // need not; the bundle flag comes from the layout).
    let (module, error) = create_shader(&device, TABLE_RENDER_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let layout = create_pipeline_layout(&device, true);

    let format = wgt::TextureFormat::Rgba8Unorm;
    let (pipeline, error) = device.create_render_pipeline(
        wgc::pipeline::RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: wgc::pipeline::VertexState {
                stage: wgc::pipeline::ProgrammableStageDescriptor {
                    module: Arc::clone(&module),
                    entry_point: Some(Cow::Borrowed("vs")),
                    constants: Default::default(),
                    zero_initialize_workgroup_memory: false,
                },
                buffers: Cow::Owned(Vec::new()),
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgc::pipeline::FragmentState {
                stage: wgc::pipeline::ProgrammableStageDescriptor {
                    module,
                    entry_point: Some(Cow::Borrowed("fs")),
                    constants: Default::default(),
                    zero_initialize_workgroup_memory: false,
                },
                targets: Cow::Owned(vec![Some(wgt::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgt::ColorWrites::ALL,
                })]),
            }),
            multiview_mask: None,
            cache: None,
        }
        .into(),
    );
    assert!(
        error.is_none(),
        "render pipeline creation failed: {error:?}"
    );

    let mut bundle_encoder = device
        .create_render_bundle_encoder(&wgc::command::RenderBundleEncoderDescriptor {
            label: None,
            color_formats: Cow::Owned(vec![Some(format)]),
            depth_stencil: None,
            sample_count: 1,
            multiview: None,
        })
        .expect("bundle encoder creation should succeed");

    bundle_encoder.set_pipeline(pipeline);

    let (_bundle, error) = with_error_scope(&device, || {
        bundle_encoder.finish(&wgc::command::RenderBundleDescriptor { label: None })
    });
    let debug = format!("{error:?}");
    assert!(
        debug.contains("ResourceTableUnsupported"),
        "expected ResourceTableUnsupported, got {debug}"
    );
}

// ---------------------------------------------------------------------------
// Work item 0.9: submit-time usage-conflict validation. A texture bound in a
// resource table that is also used, in a submission that binds the table, in a
// way that forces an image layout incompatible with sampling it (written,
// storage-read, copied, attached, …) is rejected under the M0 strict (v0)
// semantics (D3/D9). Pure bindful sampling and read-only depth use are benign.
// ---------------------------------------------------------------------------

/// Create a texture usable as both a render attachment and a sampled resource,
/// plus a default view of it. The view is thus valid both as a color attachment
/// (a writable usage) and as a resource-table member.
fn create_renderable_sampled_view(device: &Arc<Device>) -> Arc<TextureView> {
    create_texture_view(
        device,
        wgt::TextureUsages::RENDER_ATTACHMENT | wgt::TextureUsages::TEXTURE_BINDING,
    )
}

/// Record an empty render pass whose single color attachment is `color_view`,
/// binding `table` through the render-pass descriptor, then finish and submit.
/// Returns the error the submit reported, so the caller can assert whether a
/// conflict fired.
fn submit_render_pass_with_color_and_table(
    device: &Arc<Device>,
    queue: &Arc<Queue>,
    color_view: &Arc<TextureView>,
    table: &Arc<ResourceTable>,
) -> Option<wgt::error::Error> {
    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    let mut pass = encoder.begin_render_pass(wgc::command::ResolvedRenderPassDescriptor {
        label: None,
        color_attachments: Cow::Owned(vec![Some(wgc::command::RenderPassColorAttachment {
            view: Arc::clone(color_view),
            depth_slice: None,
            resolve_target: None,
            load_op: wgt::LoadOp::Clear(wgt::Color::BLACK),
            store_op: wgt::StoreOp::Store,
        })]),
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
        resource_table: Some(Arc::clone(table)),
    });
    pass.end();

    let cb = encoder.finish(&wgt::CommandBufferDescriptor { label: None });

    with_error_scope(device, || queue.submit(&[cb])).1
}

/// Submitting a command buffer that both binds a table containing a texture and
/// writes that same texture as a render-pass color target is rejected at submit
/// with a resource-table usage conflict (work item 0.9).
#[test]
fn submit_table_member_written_as_color_target_conflicts() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    // One texture used both as the table member and as the color target.
    let view = create_renderable_sampled_view(&device);
    table
        .update_slot(0, &view)
        .expect("bind member should succeed");

    let error = submit_render_pass_with_color_and_table(&device, &queue, &view, &table);

    let debug = format!("{error:?}");
    assert!(
        debug.contains("IncompatibleMemberUsage"),
        "expected a resource-table usage conflict, got {debug}"
    );
}

/// Submitting a command buffer that binds a table containing a texture and also
/// copies *from* that same texture with a top-level `copy_texture_to_buffer`
/// (`COPY_SRC`, which forces `TRANSFER_SRC_OPTIMAL`) is rejected at submit (work
/// item 0.9, finding fix). Unlike the color-target case, the copy records directly
/// on the command-buffer tracker rather than through a pass, exercising the
/// top-level-transfer collection path.
#[test]
fn submit_table_member_copied_from_conflicts() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    // A texture usable both as a sampled table member and as a copy source.
    let (texture, view) = create_texture_and_view(
        &device,
        wgt::TextureUsages::TEXTURE_BINDING | wgt::TextureUsages::COPY_SRC,
    );
    table
        .update_slot(0, &view)
        .expect("bind member should succeed");

    let (buffer, error) = device.create_buffer(&wgt::BufferDescriptor {
        label: None,
        size: 1024,
        usage: wgt::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    assert!(error.is_none(), "buffer creation failed: {error:?}");

    let encoder = device.create_command_encoder(&wgt::CommandEncoderDescriptor { label: None });

    // An empty compute pass that binds the table, so the command buffer references
    // it (and the early exit in the conflict check is passed).
    let mut pass = encoder.begin_compute_pass(&wgc::command::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_resource_table(Some(Arc::clone(&table)));
    pass.end();

    // Top-level copy *from* the member texture (records `COPY_SRC` on the command
    // buffer's own tracker).
    encoder.copy_texture_to_buffer(
        &wgt::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgt::Origin3d::ZERO,
            aspect: wgt::TextureAspect::All,
        },
        &wgt::TexelCopyBufferInfo {
            buffer,
            layout: wgt::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: Some(4),
            },
        },
        &wgt::Extent3d {
            width: 4,
            height: 4,
            depth_or_array_layers: 1,
        },
    );

    let cb = encoder.finish(&wgt::CommandBufferDescriptor { label: None });

    let (_index, error) = with_error_scope(&device, || queue.submit(&[cb]));
    let debug = format!("{error:?}");
    assert!(
        debug.contains("IncompatibleMemberUsage"),
        "expected a resource-table usage conflict for COPY_SRC, got {debug}"
    );
}

/// Positive control: writing a texture that is *not* a member of the bound table
/// (the common "render to a target while a table is bound for sampling" flow) is
/// not a conflict, even though the table is referenced and has a member.
#[test]
fn submit_written_texture_not_in_table_no_conflict() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    // The table's member is a distinct texture from the color target.
    let member_view = create_sampled_view(&device);
    table
        .update_slot(0, &member_view)
        .expect("bind member should succeed");

    let color_view = create_renderable_sampled_view(&device);

    let error = submit_render_pass_with_color_and_table(&device, &queue, &color_view, &table);
    assert!(
        error.is_none(),
        "writing a non-member texture must not conflict: {error:?}"
    );
}

/// Positive control: a table member that is only *sampled* (read) in the
/// submission — never written — is not a conflict. Here the table is bound in an
/// empty render pass with an unrelated color target; the member is not touched by
/// any writable usage, so submit succeeds and the slot stays reusable.
#[test]
fn submit_table_member_read_only_no_conflict() {
    let instance = create_noop_instance();
    let (device, queue) = request_device(&instance, true);

    let (table, error) = device.create_resource_table(&descriptor(4));
    assert!(error.is_none(), "table creation failed: {error:?}");

    let member_view = create_sampled_view(&device);
    table
        .update_slot(0, &member_view)
        .expect("bind member should succeed");

    // A separate, non-member color target keeps the pass valid without writing
    // the member.
    let color_view = create_renderable_sampled_view(&device);
    let error = submit_render_pass_with_color_and_table(&device, &queue, &color_view, &table);
    assert!(
        error.is_none(),
        "read-only member must not conflict: {error:?}"
    );

    // The member's slot is reusable after the (synchronous) submission completes.
    device
        .poll(wgt::PollType::Poll)
        .expect("poll should succeed");
    table
        .update_slot(0, &member_view)
        .expect("member slot should be reusable after completion");
}

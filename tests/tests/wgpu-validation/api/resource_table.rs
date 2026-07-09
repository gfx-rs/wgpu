//! Validation tests for the wgpu-core resource table object (work item 0.6 of
//! the bindless feature), its encoder-state plumbing (work item 0.7), and the
//! queue machinery / host slot-update flows (work item 0.8).
//!
//! These run against the `noop` backend through the `wgpu-core` [`Global`] API
//! directly (like [`limit_buckets`]), because the public `wgpu` resource-table
//! API does not exist yet (work item 0.11).
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
//! [`Global`]: wgpu_core::global::Global
//! [`limit_buckets`]: crate::limit_buckets

use std::borrow::Cow;

use wgpu_core as wgc;
use wgpu_types as wgt;

use wgc::binding_model::CreatePipelineLayoutError;
use wgc::command::CommandEncoderError;
use wgc::pipeline::{
    CreateComputePipelineError, CreateShaderModuleError, ResourceTablePipelineError,
};
use wgc::resource_table::{
    CreateResourceTableError, UpdateResourceTableError, MAX_RESOURCE_TABLE_SIZE,
};

fn create_noop_global() -> wgc::global::Global {
    wgc::global::Global::new(
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

fn noop_adapter(global: &wgc::global::Global) -> wgc::id::AdapterId {
    global
        .request_adapter(
            &wgt::RequestAdapterOptions::default(),
            wgt::Backends::NOOP,
            None,
        )
        .expect("noop adapter should be available")
}

/// Request a device with the given required features enabled.
///
/// Any resource-table feature is experimental, so the unsafe experimental gate
/// is passed whenever `features` is non-empty (all features requested by these
/// tests are experimental resource-table bits).
fn request_device_features(
    global: &wgc::global::Global,
    features: wgt::Features,
) -> wgc::id::DeviceId {
    let adapter_id = noop_adapter(global);

    let experimental_features = if features.is_empty() {
        wgt::ExperimentalFeatures::disabled()
    } else {
        // SAFETY: This is a test; the noop backend has no actual unsafe
        // behavior behind the experimental gate.
        unsafe { wgt::ExperimentalFeatures::enabled() }
    };

    let (device_id, _queue_id) = global
        .adapter_request_device(
            adapter_id,
            &wgt::DeviceDescriptor {
                required_features: features,
                experimental_features,
                ..Default::default()
            },
            None,
            None,
        )
        .expect("device creation should succeed");
    device_id
}

/// Request a device, optionally enabling the sampling resource table feature.
fn request_device(global: &wgc::global::Global, with_feature: bool) -> wgc::id::DeviceId {
    let features = if with_feature {
        wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE
    } else {
        wgt::Features::empty()
    };
    request_device_features(global, features)
}

fn descriptor(size: u32) -> wgc::resource_table::ResourceTableDescriptor<'static> {
    wgt::ResourceTableDescriptor { label: None, size }
}

/// Request a device and its queue with the sampling resource table feature.
fn request_device_and_queue(global: &wgc::global::Global) -> (wgc::id::DeviceId, wgc::id::QueueId) {
    let adapter_id = noop_adapter(global);
    global
        .adapter_request_device(
            adapter_id,
            &wgt::DeviceDescriptor {
                required_features: wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE,
                // SAFETY: test-only; the noop backend has no unsafe behavior.
                experimental_features: unsafe { wgt::ExperimentalFeatures::enabled() },
                ..Default::default()
            },
            None,
            None,
        )
        .expect("device creation should succeed")
}

/// Create a texture with the given usage and a default view of it. Returns both
/// the texture and view ids.
fn create_texture_and_view(
    global: &wgc::global::Global,
    device_id: wgc::id::DeviceId,
    usage: wgt::TextureUsages,
) -> (wgc::id::TextureId, wgc::id::TextureViewId) {
    let (texture_id, error) = global.device_create_texture(
        device_id,
        &wgt::TextureDescriptor {
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
        },
        None,
    );
    assert!(error.is_none(), "texture creation failed: {error:?}");

    let (view_id, error) = global.texture_create_view(
        texture_id,
        &wgc::resource::TextureViewDescriptor {
            label: None,
            format: None,
            dimension: None,
            usage: None,
            range: wgt::ImageSubresourceRange::default(),
        },
        None,
    );
    assert!(error.is_none(), "texture view creation failed: {error:?}");
    (texture_id, view_id)
}

/// A texture view with the given usage.
fn create_texture_view(
    global: &wgc::global::Global,
    device_id: wgc::id::DeviceId,
    usage: wgt::TextureUsages,
) -> wgc::id::TextureViewId {
    create_texture_and_view(global, device_id, usage).1
}

/// A sampled (`TEXTURE_BINDING`) texture view, valid for binding into a table.
fn create_sampled_view(
    global: &wgc::global::Global,
    device_id: wgc::id::DeviceId,
) -> wgc::id::TextureViewId {
    create_texture_view(global, device_id, wgt::TextureUsages::TEXTURE_BINDING)
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
    global: &wgc::global::Global,
    device_id: wgc::id::DeviceId,
    wgsl: &str,
) -> (wgc::id::ShaderModuleId, Option<CreateShaderModuleError>) {
    global.device_create_shader_module(
        device_id,
        &wgc::pipeline::ShaderModuleDescriptor {
            label: None,
            runtime_checks: wgt::ShaderRuntimeChecks::default(),
        },
        wgc::pipeline::ShaderModuleSource::Wgsl(Cow::Borrowed(wgsl)),
        None,
    )
}

fn create_pipeline_layout(
    global: &wgc::global::Global,
    device_id: wgc::id::DeviceId,
    uses_resource_table: bool,
) -> (wgc::id::PipelineLayoutId, Option<CreatePipelineLayoutError>) {
    global.device_create_pipeline_layout(
        device_id,
        &pipeline_layout_desc(uses_resource_table),
        None,
    )
}

fn create_compute_pipeline(
    global: &wgc::global::Global,
    device_id: wgc::id::DeviceId,
    layout: Option<wgc::id::PipelineLayoutId>,
    module: wgc::id::ShaderModuleId,
) -> (
    wgc::id::ComputePipelineId,
    Option<CreateComputePipelineError>,
) {
    global.device_create_compute_pipeline(
        device_id,
        &wgc::pipeline::ComputePipelineDescriptor {
            label: None,
            layout,
            stage: wgc::pipeline::ProgrammableStageDescriptor {
                module,
                entry_point: Some(Cow::Borrowed("main")),
                constants: Default::default(),
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        },
        None,
    )
}

/// Creating a resource table without the feature enabled fails with
/// `MissingFeatures`.
#[test]
fn create_without_feature_fails() {
    let global = create_noop_global();
    let device_id = request_device(&global, false);

    let (_id, error) = global.device_create_resource_table(device_id, &descriptor(8), None);

    assert!(
        matches!(error, Some(CreateResourceTableError::MissingFeatures(_))),
        "expected MissingFeatures, got {error:?}"
    );
}

/// Creating a resource table with the feature enabled succeeds.
#[test]
fn create_with_feature_succeeds() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (id, error) = global.device_create_resource_table(device_id, &descriptor(8), None);

    assert!(error.is_none(), "unexpected error: {error:?}");

    // Cleanup: destroy and drop should both be no-panic.
    global.resource_table_destroy(id);
    global.resource_table_drop(id);
}

/// A zero-slot resource table is rejected.
#[test]
fn create_zero_size_fails() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (_id, error) = global.device_create_resource_table(device_id, &descriptor(0), None);

    assert!(
        matches!(error, Some(CreateResourceTableError::ZeroSize)),
        "expected ZeroSize, got {error:?}"
    );
}

/// A resource table with more than `MAX_RESOURCE_TABLE_SIZE` slots is rejected,
/// while exactly `MAX_RESOURCE_TABLE_SIZE` is allowed.
#[test]
fn create_size_bounds() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    // The maximum is allowed.
    let (max_id, error) =
        global.device_create_resource_table(device_id, &descriptor(MAX_RESOURCE_TABLE_SIZE), None);
    assert!(error.is_none(), "max size should be allowed, got {error:?}");
    global.resource_table_drop(max_id);

    // One past the maximum is rejected.
    let (_id, error) = global.device_create_resource_table(
        device_id,
        &descriptor(MAX_RESOURCE_TABLE_SIZE + 1),
        None,
    );
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
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "unexpected error: {error:?}");

    global.resource_table_destroy(id);
    // Destroying again must not panic.
    global.resource_table_destroy(id);
    // Dropping after destroy must not panic.
    global.resource_table_drop(id);
}

/// Dropping a resource table without destroying it first is fine (drop-based
/// cleanup path).
#[test]
fn drop_without_destroy() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "unexpected error: {error:?}");

    global.resource_table_drop(id);
}

// ---------------------------------------------------------------------------
// Work item 0.7: pipeline-layout plumbing, shader/pipeline validation, and
// pass-scope `set_resource_table` with draw/dispatch-time checks.
// ---------------------------------------------------------------------------

/// Creating a pipeline layout with `uses_resource_table` set requires the
/// sampling resource table feature.
#[test]
fn pipeline_layout_flag_requires_feature() {
    let global = create_noop_global();
    let device_id = request_device_features(&global, wgt::Features::empty());

    let (_id, error) = create_pipeline_layout(&global, device_id, true);

    assert!(
        matches!(error, Some(CreatePipelineLayoutError::MissingFeatures(_))),
        "expected MissingFeatures, got {error:?}"
    );
}

/// With the feature enabled, a `uses_resource_table` pipeline layout is created
/// successfully.
#[test]
fn pipeline_layout_flag_with_feature_succeeds() {
    let global = create_noop_global();
    let device_id =
        request_device_features(&global, wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE);

    let (_id, error) = create_pipeline_layout(&global, device_id, true);

    assert!(error.is_none(), "unexpected error: {error:?}");
}

/// A shader that uses `getResource` fails to validate without the sampling
/// feature enabled (the `RESOURCE_TABLE` naga capability is gated on it).
#[test]
fn table_shader_requires_feature() {
    let global = create_noop_global();
    let device_id = request_device_features(&global, wgt::Features::empty());

    let (_id, error) = create_shader(&global, device_id, TABLE_COMPUTE_WGSL);

    assert!(
        error.is_some(),
        "expected a shader validation error without the feature"
    );
}

/// A pipeline whose shader uses `getResource` must be created with a layout that
/// declares `uses_resource_table`; otherwise pipeline creation fails.
#[test]
fn table_shader_requires_table_layout() {
    let global = create_noop_global();
    let device_id = request_device_features(&global, sampling_and_unchecked());

    let (module, error) = create_shader(&global, device_id, TABLE_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    // A layout that does *not* declare a resource table.
    let (layout, error) = create_pipeline_layout(&global, device_id, false);
    assert!(error.is_none(), "layout creation failed: {error:?}");

    let (_pipeline, error) = create_compute_pipeline(&global, device_id, Some(layout), module);

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
    let global = create_noop_global();
    // Sampling only: enough to create the layout and validate the shader, but
    // not to create the (unchecked-only) pipeline.
    let device_id =
        request_device_features(&global, wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE);

    let (module, error) = create_shader(&global, device_id, TABLE_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let (layout, error) = create_pipeline_layout(&global, device_id, true);
    assert!(error.is_none(), "layout creation failed: {error:?}");

    let (_pipeline, error) = create_compute_pipeline(&global, device_id, Some(layout), module);

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
    let global = create_noop_global();
    let device_id = request_device_features(&global, sampling_and_unchecked());

    let (module, error) = create_shader(&global, device_id, TABLE_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let (layout, error) = create_pipeline_layout(&global, device_id, true);
    assert!(error.is_none(), "layout creation failed: {error:?}");

    let (_pipeline, error) = create_compute_pipeline(&global, device_id, Some(layout), module);

    assert!(error.is_none(), "expected success, got {error:?}");
}

/// Dispatching with a pipeline whose layout declares a resource table, without
/// binding one via `set_resource_table`, is a validation error (surfaced at
/// `command_encoder_finish`). The shader itself need not use `getResource`: the
/// check is on the layout flag, because the table's descriptor set is reserved
/// at the highest set index regardless.
#[test]
fn dispatch_without_bound_table_fails() {
    let global = create_noop_global();
    let device_id =
        request_device_features(&global, wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE);

    let (module, error) = create_shader(&global, device_id, TRIVIAL_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let (layout, error) = create_pipeline_layout(&global, device_id, true);
    assert!(error.is_none(), "layout creation failed: {error:?}");

    let (pipeline, error) = create_compute_pipeline(&global, device_id, Some(layout), module);
    assert!(error.is_none(), "pipeline creation failed: {error:?}");

    let (encoder_id, error) = global.device_create_command_encoder(
        device_id,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    );
    assert!(error.is_none(), "encoder creation failed: {error:?}");

    let (mut pass, error) = global.command_encoder_begin_compute_pass(
        encoder_id,
        &wgc::command::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        },
    );
    assert!(error.is_none(), "begin compute pass failed: {error:?}");

    global
        .compute_pass_set_pipeline(&mut pass, pipeline)
        .expect("set_pipeline should record");
    global
        .compute_pass_dispatch_workgroups(&mut pass, 1, 1, 1)
        .expect("dispatch should record");
    global
        .compute_pass_end(&mut pass)
        .expect("compute_pass_end should record");

    let (_cb, finish_error) = global.command_encoder_finish(
        encoder_id,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    );

    let (_label, error) = finish_error.expect("expected a finish error");
    assert!(
        matches!(error, CommandEncoderError::ComputePass(_)),
        "expected a compute-pass error, got {error:?}"
    );
    let debug = format!("{error:?}");
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
    let global = create_noop_global();
    let device_id =
        request_device_features(&global, wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE);

    let (module, error) = create_shader(&global, device_id, TRIVIAL_COMPUTE_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let (layout, error) = create_pipeline_layout(&global, device_id, true);
    assert!(error.is_none(), "layout creation failed: {error:?}");

    let (pipeline, error) = create_compute_pipeline(&global, device_id, Some(layout), module);
    assert!(error.is_none(), "pipeline creation failed: {error:?}");

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let (encoder_id, error) = global.device_create_command_encoder(
        device_id,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    );
    assert!(error.is_none(), "encoder creation failed: {error:?}");

    let (mut pass, error) = global.command_encoder_begin_compute_pass(
        encoder_id,
        &wgc::command::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        },
    );
    assert!(error.is_none(), "begin compute pass failed: {error:?}");

    global
        .compute_pass_set_resource_table(&mut pass, Some(table_id))
        .expect("set_resource_table should record");
    global
        .compute_pass_set_pipeline(&mut pass, pipeline)
        .expect("set_pipeline should record");
    global
        .compute_pass_dispatch_workgroups(&mut pass, 1, 1, 1)
        .expect("dispatch should record");
    global
        .compute_pass_end(&mut pass)
        .expect("compute_pass_end should record");

    let (_cb, finish_error) = global.command_encoder_finish(
        encoder_id,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    );

    assert!(
        finish_error.is_none(),
        "expected finish to succeed, got {finish_error:?}"
    );
}

/// Binding a resource table created on a different device to a pass is a
/// validation error (surfaced at `command_encoder_finish`).
#[test]
fn resource_table_wrong_device_fails() {
    let global = create_noop_global();
    let device_a =
        request_device_features(&global, wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE);
    let device_b =
        request_device_features(&global, wgt::Features::EXPERIMENTAL_SAMPLING_RESOURCE_TABLE);

    // Table on device A.
    let (table_id, error) = global.device_create_resource_table(device_a, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    // Encoder / pass on device B.
    let (encoder_id, error) = global.device_create_command_encoder(
        device_b,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    );
    assert!(error.is_none(), "encoder creation failed: {error:?}");

    let (mut pass, error) = global.command_encoder_begin_compute_pass(
        encoder_id,
        &wgc::command::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        },
    );
    assert!(error.is_none(), "begin compute pass failed: {error:?}");

    global
        .compute_pass_set_resource_table(&mut pass, Some(table_id))
        .expect("set_resource_table should record");
    global
        .compute_pass_end(&mut pass)
        .expect("compute_pass_end should record");

    let (_cb, finish_error) = global.command_encoder_finish(
        encoder_id,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    );

    let (_label, error) = finish_error.expect("expected a finish error");
    assert!(
        matches!(error, CommandEncoderError::ComputePass(_)),
        "expected a compute-pass error, got {error:?}"
    );
    let debug = format!("{error:?}");
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
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_id);

    global
        .resource_table_update(table_id, 0, view_id)
        .expect("update should succeed");
    // Overwriting a still-available slot is fine.
    global
        .resource_table_update(table_id, 3, view_id)
        .expect("update should succeed");
}

/// Updating a slot outside the table's range fails with `SlotOutOfBounds`.
#[test]
fn update_slot_out_of_bounds() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_id);

    let result = global.resource_table_update(table_id, 4, view_id);
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
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    // Renderable but not sampleable: has a view but no `TEXTURE_BINDING`.
    let view_id = create_texture_view(
        &global,
        device_id,
        wgt::TextureUsages::RENDER_ATTACHMENT | wgt::TextureUsages::COPY_DST,
    );

    let result = global.resource_table_update(table_id, 0, view_id);
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
    let global = create_noop_global();
    let device_a = request_device(&global, true);
    let device_b = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_a, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_b);

    let result = global.resource_table_update(table_id, 0, view_id);
    assert!(
        matches!(result, Err(UpdateResourceTableError::Device(_))),
        "expected a device error, got {result:?}"
    );
}

/// Updating a slot of a destroyed table fails with `DestroyedResource`.
#[test]
fn update_destroyed_table() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_id);

    global.resource_table_destroy(table_id);

    let result = global.resource_table_update(table_id, 0, view_id);
    assert!(
        matches!(result, Err(UpdateResourceTableError::DestroyedResource(_))),
        "expected DestroyedResource, got {result:?}"
    );
}

/// Binding a view whose parent texture has been destroyed fails with
/// `DestroyedResource`.
#[test]
fn update_with_destroyed_texture() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let (texture_id, view_id) =
        create_texture_and_view(&global, device_id, wgt::TextureUsages::TEXTURE_BINDING);

    global.texture_destroy(texture_id);

    let result = global.resource_table_update(table_id, 0, view_id);
    assert!(
        matches!(result, Err(UpdateResourceTableError::DestroyedResource(_))),
        "expected DestroyedResource, got {result:?}"
    );
}

/// `insert_binding` assigns the lowest-available slot (D8), and reports
/// `NoAvailableSlot` once the table is full.
#[test]
fn insert_binding_assigns_lowest_available() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(2), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_id);

    let slot0 = global
        .resource_table_insert_binding(table_id, view_id)
        .expect("first insert should succeed");
    assert_eq!(slot0, 0);

    let slot1 = global
        .resource_table_insert_binding(table_id, view_id)
        .expect("second insert should succeed");
    assert_eq!(slot1, 1);

    // Table is now full.
    let result = global.resource_table_insert_binding(table_id, view_id);
    assert!(
        matches!(result, Err(UpdateResourceTableError::NoAvailableSlot)),
        "expected NoAvailableSlot, got {result:?}"
    );
}

/// Removing a binding frees its slot so the next `insert_binding` reuses it.
#[test]
fn remove_binding_frees_slot() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(2), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_id);

    assert_eq!(
        global
            .resource_table_insert_binding(table_id, view_id)
            .unwrap(),
        0
    );
    assert_eq!(
        global
            .resource_table_insert_binding(table_id, view_id)
            .unwrap(),
        1
    );

    global
        .resource_table_remove_binding(table_id, 0)
        .expect("remove should succeed");

    // The freed slot is the lowest available again.
    assert_eq!(
        global
            .resource_table_insert_binding(table_id, view_id)
            .unwrap(),
        0
    );
}

/// Removing an out-of-bounds slot is a validation error.
#[test]
fn remove_binding_out_of_bounds() {
    let global = create_noop_global();
    let device_id = request_device(&global, true);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(2), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let result = global.resource_table_remove_binding(table_id, 2);
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
    let global = create_noop_global();
    let (device_id, queue_id) = request_device_and_queue(&global);

    let (table_id, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table creation failed: {error:?}");

    let view_id = create_sampled_view(&global, device_id);
    global
        .resource_table_update(table_id, 0, view_id)
        .expect("update should succeed");

    let (encoder_id, error) = global.device_create_command_encoder(
        device_id,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    );
    assert!(error.is_none(), "encoder creation failed: {error:?}");

    let (mut pass, error) = global.command_encoder_begin_compute_pass(
        encoder_id,
        &wgc::command::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        },
    );
    assert!(error.is_none(), "begin compute pass failed: {error:?}");

    global
        .compute_pass_set_resource_table(&mut pass, Some(table_id))
        .expect("set_resource_table should record");
    global
        .compute_pass_end(&mut pass)
        .expect("compute_pass_end should record");

    let (cb_id, finish_error) = global.command_encoder_finish(
        encoder_id,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    );
    assert!(finish_error.is_none(), "finish failed: {finish_error:?}");

    global
        .queue_submit(queue_id, &[cb_id])
        .expect("submit should succeed");

    // The noop submission has completed; the slot is reusable.
    global
        .device_poll(device_id, wgt::PollType::Poll)
        .expect("poll should succeed");
    global
        .resource_table_update(table_id, 0, view_id)
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
    let global = create_noop_global();
    let (device_id, queue_id) = request_device_and_queue(&global);

    let (table_a, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table A creation failed: {error:?}");
    let (table_b, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table B creation failed: {error:?}");

    // Distinct member textures so each table's gap has a barrier to compute.
    let view_a = create_sampled_view(&global, device_id);
    let view_b = create_sampled_view(&global, device_id);
    global
        .resource_table_update(table_a, 0, view_a)
        .expect("update A should succeed");
    global
        .resource_table_update(table_b, 0, view_b)
        .expect("update B should succeed");

    let (encoder_id, error) = global.device_create_command_encoder(
        device_id,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    );
    assert!(error.is_none(), "encoder creation failed: {error:?}");

    let (mut pass, error) = global.command_encoder_begin_compute_pass(
        encoder_id,
        &wgc::command::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        },
    );
    assert!(error.is_none(), "begin compute pass failed: {error:?}");

    // Bind A, then rebind B within the same pass. Before finding M1 only B (the
    // final binding) would have had a gap recorded.
    global
        .compute_pass_set_resource_table(&mut pass, Some(table_a))
        .expect("set_resource_table A should record");
    global
        .compute_pass_set_resource_table(&mut pass, Some(table_b))
        .expect("set_resource_table B should record");
    global
        .compute_pass_end(&mut pass)
        .expect("compute_pass_end should record");

    let (cb_id, finish_error) = global.command_encoder_finish(
        encoder_id,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    );
    assert!(finish_error.is_none(), "finish failed: {finish_error:?}");

    global
        .queue_submit(queue_id, &[cb_id])
        .expect("submit should succeed");
    global
        .device_poll(device_id, wgt::PollType::Poll)
        .expect("poll should succeed");

    // Both tables were marked and their gaps spliced; both slots are reusable
    // once the submission completes.
    global
        .resource_table_update(table_a, 0, view_a)
        .expect("table A slot should be reusable after completion");
    global
        .resource_table_update(table_b, 0, view_b)
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
    let global = create_noop_global();
    let (device_id, queue_id) = request_device_and_queue(&global);

    let (table1, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table 1 creation failed: {error:?}");
    let (table2, error) = global.device_create_resource_table(device_id, &descriptor(4), None);
    assert!(error.is_none(), "table 2 creation failed: {error:?}");

    // One shared texture/view bound into both tables.
    let view_id = create_sampled_view(&global, device_id);
    global
        .resource_table_update(table1, 0, view_id)
        .expect("update table 1 should succeed");
    global
        .resource_table_update(table2, 0, view_id)
        .expect("update table 2 should succeed");

    let (encoder_id, error) = global.device_create_command_encoder(
        device_id,
        &wgt::CommandEncoderDescriptor { label: None },
        None,
    );
    assert!(error.is_none(), "encoder creation failed: {error:?}");

    // Pass 1 binds table1; pass 2 binds table2 — both reference the same texture.
    for table in [table1, table2] {
        let (mut pass, error) = global.command_encoder_begin_compute_pass(
            encoder_id,
            &wgc::command::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            },
        );
        assert!(error.is_none(), "begin compute pass failed: {error:?}");
        global
            .compute_pass_set_resource_table(&mut pass, Some(table))
            .expect("set_resource_table should record");
        global
            .compute_pass_end(&mut pass)
            .expect("compute_pass_end should record");
    }

    let (cb_id, finish_error) = global.command_encoder_finish(
        encoder_id,
        &wgt::CommandBufferDescriptor { label: None },
        None,
    );
    assert!(finish_error.is_none(), "finish failed: {finish_error:?}");

    global
        .queue_submit(queue_id, &[cb_id])
        .expect("submit should succeed");
    global
        .device_poll(device_id, wgt::PollType::Poll)
        .expect("poll should succeed");

    global
        .resource_table_update(table1, 0, view_id)
        .expect("table 1 slot should be reusable after completion");
    global
        .resource_table_update(table2, 0, view_id)
        .expect("table 2 slot should be reusable after completion");
}

/// A render bundle that records a pipeline whose layout declares a resource
/// table is rejected at bundle creation in M0 (2026-07-09 user decision).
#[test]
fn table_using_render_bundle_rejected() {
    let global = create_noop_global();
    let device_id = request_device_features(&global, sampling_and_unchecked());

    // A trivial render pipeline whose *layout* uses a resource table (its shader
    // need not; the bundle flag comes from the layout).
    let (module, error) = create_shader(&global, device_id, TABLE_RENDER_WGSL);
    assert!(error.is_none(), "shader should compile: {error:?}");

    let (layout, error) = create_pipeline_layout(&global, device_id, true);
    assert!(error.is_none(), "layout creation failed: {error:?}");

    let format = wgt::TextureFormat::Rgba8Unorm;
    let (pipeline, error) = global.device_create_render_pipeline(
        device_id,
        &wgc::pipeline::RenderPipelineDescriptor {
            label: None,
            layout: Some(layout),
            vertex: wgc::pipeline::VertexState {
                stage: wgc::pipeline::ProgrammableStageDescriptor {
                    module,
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
        },
        None,
    );
    assert!(
        error.is_none(),
        "render pipeline creation failed: {error:?}"
    );

    let (mut bundle_encoder, error) = global.device_create_render_bundle_encoder(
        device_id,
        &wgc::command::RenderBundleEncoderDescriptor {
            label: None,
            color_formats: Cow::Owned(vec![Some(format)]),
            depth_stencil: None,
            sample_count: 1,
            multiview: None,
        },
    );
    assert!(error.is_none(), "bundle encoder creation failed: {error:?}");

    global
        .render_bundle_encoder_set_pipeline(&mut bundle_encoder, pipeline)
        .expect("set_pipeline should record");

    let (_bundle_id, finish_error) = global.render_bundle_encoder_finish(
        &mut bundle_encoder,
        &wgc::command::RenderBundleDescriptor { label: None },
        None,
    );

    let error = finish_error.expect("expected a bundle finish error");
    let debug = format!("{error:?}");
    assert!(
        debug.contains("ResourceTableUnsupported"),
        "expected ResourceTableUnsupported, got {debug}"
    );
}

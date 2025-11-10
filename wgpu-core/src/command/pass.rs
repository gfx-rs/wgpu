//! Generic pass functions that both compute and render passes need.

use crate::binding_model::{BindError, BindGroup, PushConstantUploadError};
use crate::command::bind::Binder;
use crate::command::encoder::EncodingState;
use crate::command::memory_init::SurfacesInDiscardState;
use crate::command::{DebugGroupError, QueryResetMap, QueryUseError};
use crate::device::{Device, DeviceError, MissingFeatures};
use crate::pipeline::LateSizedBufferGroup;
use crate::resource::{DestroyedResourceError, Labeled, ParentDevice, QuerySet};
use crate::track::{ResourceUsageCompatibilityError, UsageScope};
use crate::{api_log, binding_model};
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::str;
use thiserror::Error;
use wgt::error::{ErrorType, WebGpuError};
use wgt::DynamicOffset;

#[derive(Clone, Debug, Error)]
#[error(
    "Bind group index {index} is greater than the device's requested `max_bind_group` limit {max}"
)]
pub struct BindGroupIndexOutOfRange {
    pub index: u32,
    pub max: u32,
}

#[derive(Clone, Debug, Error)]
#[error("Pipeline must be set")]
pub struct MissingPipeline;

#[derive(Clone, Debug, Error)]
#[error("Setting `values_offset` to be `None` is only for internal use in render bundles")]
pub struct InvalidValuesOffset;

impl WebGpuError for InvalidValuesOffset {
    fn webgpu_error_type(&self) -> ErrorType {
        ErrorType::Validation
    }
}

pub(crate) struct PassState<'scope, 'snatch_guard, 'cmd_enc> {
    pub(crate) base: EncodingState<'snatch_guard, 'cmd_enc>,

    /// Immediate texture inits required because of prior discards. Need to
    /// be inserted before texture reads.
    pub(crate) pending_discard_init_fixups: SurfacesInDiscardState,

    pub(crate) scope: UsageScope<'scope>,

    pub(crate) binder: Binder,

    pub(crate) temp_offsets: Vec<u32>,

    pub(crate) dynamic_offset_count: usize,

    pub(crate) string_offset: usize,
}

pub(crate) fn set_bind_group<E>(
    state: &mut PassState,
    device: &Arc<Device>,
    dynamic_offsets: &[DynamicOffset],
    index: u32,
    num_dynamic_offsets: usize,
    bind_group: Option<Arc<BindGroup>>,
    merge_bind_groups: bool,
) -> Result<(), E>
where
    E: From<DeviceError>
        + From<BindGroupIndexOutOfRange>
        + From<ResourceUsageCompatibilityError>
        + From<DestroyedResourceError>
        + From<BindError>,
{
    if bind_group.is_none() {
        api_log!("Pass::set_bind_group {index} None");
    } else {
        api_log!(
            "Pass::set_bind_group {index} {}",
            bind_group.as_ref().unwrap().error_ident()
        );
    }

    let max_bind_groups = state.base.device.limits.max_bind_groups;
    if index >= max_bind_groups {
        return Err(BindGroupIndexOutOfRange {
            index,
            max: max_bind_groups,
        }
        .into());
    }

    state.temp_offsets.clear();
    state.temp_offsets.extend_from_slice(
        &dynamic_offsets
            [state.dynamic_offset_count..state.dynamic_offset_count + num_dynamic_offsets],
    );
    state.dynamic_offset_count += num_dynamic_offsets;

    let Some(bind_group) = bind_group else {
        // TODO: Handle bind_group None.
        return Ok(());
    };

    // Add the bind group to the tracker. This is done for both compute and
    // render passes, and is used to fail submission of the command buffer if
    // any resource in any of the bind groups has been destroyed, whether or
    // not the bind group is actually used by the pipeline.
    let bind_group = state.base.tracker.bind_groups.insert_single(bind_group);

    bind_group.same_device(device)?;

    bind_group.validate_dynamic_bindings(index, &state.temp_offsets)?;

    if merge_bind_groups {
        // Merge the bind group's resources into the tracker. We only do this
        // for render passes. For compute passes it is done per dispatch in
        // [`flush_bindings`].
        unsafe {
            state.scope.merge_bind_group(&bind_group.used)?;
        }
    }
    //Note: stateless trackers are not merged: the lifetime reference
    // is held to the bind group itself.

    state
        .binder
        .assign_group(index as usize, bind_group, &state.temp_offsets);

    Ok(())
}

/// Implementation of `flush_bindings` for both compute and render passes.
///
/// See the compute pass version of `State::flush_bindings` for an explanation
/// of some differences in handling the two types of passes.
pub(super) fn flush_bindings_helper(state: &mut PassState) -> Result<(), DestroyedResourceError> {
    let range = state.binder.take_rebind_range();
    if range.is_empty() {
        return Ok(());
    }

    let entries = state.binder.entries(range);

    for (_, entry) in entries.clone() {
        let bind_group = entry.group.as_ref().unwrap();

        state.base.buffer_memory_init_actions.extend(
            bind_group.used_buffer_ranges.iter().filter_map(|action| {
                action
                    .buffer
                    .initialization_status
                    .read()
                    .check_action(action)
            }),
        );
        for action in bind_group.used_texture_ranges.iter() {
            state.pending_discard_init_fixups.extend(
                state
                    .base
                    .texture_memory_actions
                    .register_init_action(action),
            );
        }

        let used_resource = bind_group
            .used
            .acceleration_structures
            .into_iter()
            .map(|tlas| crate::ray_tracing::AsAction::UseTlas(tlas.clone()));

        state.base.as_actions.extend(used_resource);
    }

    if let Some(pipeline_layout) = state.binder.pipeline_layout.as_ref() {
        for (i, e) in entries {
            if let Some(group) = e.group.as_ref() {
                let raw_bg = group.try_raw(state.base.snatch_guard)?;
                unsafe {
                    state.base.raw_encoder.set_bind_group(
                        pipeline_layout.raw(),
                        i as u32,
                        Some(raw_bg),
                        &e.dynamic_offsets,
                    );
                }
            }
        }
    }

    Ok(())
}

pub(super) fn change_pipeline_layout<E, F: FnOnce()>(
    state: &mut PassState,
    pipeline_layout: &Arc<binding_model::PipelineLayout>,
    late_sized_buffer_groups: &[LateSizedBufferGroup],
    f: F,
) -> Result<(), E>
where
    E: From<DestroyedResourceError>,
{
    if state.binder.pipeline_layout.is_none()
        || !state
            .binder
            .pipeline_layout
            .as_ref()
            .unwrap()
            .is_equal(pipeline_layout)
    {
        state
            .binder
            .change_pipeline_layout(pipeline_layout, late_sized_buffer_groups);

        f();

        let non_overlapping =
            super::bind::compute_nonoverlapping_ranges(&pipeline_layout.push_constant_ranges);

        // Clear push constant ranges
        for range in non_overlapping {
            let offset = range.range.start;
            let size_bytes = range.range.end - offset;
            super::push_constant_clear(offset, size_bytes, |clear_offset, clear_data| unsafe {
                state.base.raw_encoder.set_push_constants(
                    pipeline_layout.raw(),
                    range.stages,
                    clear_offset,
                    clear_data,
                );
            });
        }
    }
    Ok(())
}

pub(crate) fn set_push_constant<E, F: FnOnce(&[u32])>(
    state: &mut PassState,
    push_constant_data: &[u32],
    stages: wgt::ShaderStages,
    offset: u32,
    size_bytes: u32,
    values_offset: Option<u32>,
    f: F,
) -> Result<(), E>
where
    E: From<PushConstantUploadError> + From<InvalidValuesOffset> + From<MissingPipeline>,
{
    api_log!("Pass::set_push_constants");

    let values_offset = values_offset.ok_or(InvalidValuesOffset)?;

    let end_offset_bytes = offset + size_bytes;
    let values_end_offset = (values_offset + size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT) as usize;
    let data_slice = &push_constant_data[(values_offset as usize)..values_end_offset];

    let pipeline_layout = state
        .binder
        .pipeline_layout
        .as_ref()
        .ok_or(MissingPipeline)?;

    pipeline_layout.validate_push_constant_ranges(stages, offset, end_offset_bytes)?;

    f(data_slice);

    unsafe {
        state
            .base
            .raw_encoder
            .set_push_constants(pipeline_layout.raw(), stages, offset, data_slice)
    }
    Ok(())
}

pub(crate) fn write_timestamp<E>(
    state: &mut PassState,
    device: &Arc<Device>,
    pending_query_resets: Option<&mut QueryResetMap>,
    query_set: Arc<QuerySet>,
    query_index: u32,
) -> Result<(), E>
where
    E: From<MissingFeatures> + From<QueryUseError> + From<DeviceError>,
{
    api_log!(
        "Pass::write_timestamps {query_index} {}",
        query_set.error_ident()
    );

    query_set.same_device(device)?;

    state
        .base
        .device
        .require_features(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES)?;

    let query_set = state.base.tracker.query_sets.insert_single(query_set);

    query_set.validate_and_write_timestamp(
        state.base.raw_encoder,
        query_index,
        pending_query_resets,
    )?;
    Ok(())
}

pub(crate) fn push_debug_group(state: &mut PassState, string_data: &[u8], len: usize) {
    *state.base.debug_scope_depth += 1;
    if !state
        .base
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        let label =
            str::from_utf8(&string_data[state.string_offset..state.string_offset + len]).unwrap();

        api_log!("Pass::push_debug_group {label:?}");
        unsafe {
            state.base.raw_encoder.begin_debug_marker(label);
        }
    }
    state.string_offset += len;
}

pub(crate) fn pop_debug_group<E>(state: &mut PassState) -> Result<(), E>
where
    E: From<DebugGroupError>,
{
    api_log!("Pass::pop_debug_group");

    if *state.base.debug_scope_depth == 0 {
        return Err(DebugGroupError::InvalidPop.into());
    }
    *state.base.debug_scope_depth -= 1;
    if !state
        .base
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        unsafe {
            state.base.raw_encoder.end_debug_marker();
        }
    }
    Ok(())
}

pub(crate) fn insert_debug_marker(state: &mut PassState, string_data: &[u8], len: usize) {
    if !state
        .base
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        let label =
            str::from_utf8(&string_data[state.string_offset..state.string_offset + len]).unwrap();
        api_log!("Pass::insert_debug_marker {label:?}");
        unsafe {
            state.base.raw_encoder.insert_debug_marker(label);
        }
    }
    state.string_offset += len;
}

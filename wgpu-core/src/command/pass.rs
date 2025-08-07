//! Generic pass functions that both compute and render passes need.

use crate::binding_model::{BindError, BindGroup, PushConstantUploadError};
use crate::command::bind::Binder;
use crate::command::memory_init::{CommandBufferTextureMemoryActions, SurfacesInDiscardState};
use crate::command::{CommandEncoder, DebugGroupError, QueryResetMap, QueryUseError};
use crate::device::{Device, DeviceError, MissingFeatures};
use crate::init_tracker::BufferInitTrackerAction;
use crate::pipeline::LateSizedBufferGroup;
use crate::ray_tracing::AsAction;
use crate::resource::{DestroyedResourceError, Labeled, ParentDevice, QuerySet};
use crate::snatch::SnatchGuard;
use crate::track::{ResourceUsageCompatibilityError, Tracker, UsageScope};
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

pub(crate) struct BaseState<'scope, 'snatch_guard, 'cmd_enc, 'raw_encoder> {
    pub(crate) device: &'cmd_enc Arc<Device>,

    pub(crate) raw_encoder: &'raw_encoder mut dyn hal::DynCommandEncoder,

    pub(crate) tracker: &'cmd_enc mut Tracker,
    pub(crate) buffer_memory_init_actions: &'cmd_enc mut Vec<BufferInitTrackerAction>,
    pub(crate) texture_memory_actions: &'cmd_enc mut CommandBufferTextureMemoryActions,
    pub(crate) as_actions: &'cmd_enc mut Vec<AsAction>,

    /// Immediate texture inits required because of prior discards. Need to
    /// be inserted before texture reads.
    pub(crate) pending_discard_init_fixups: SurfacesInDiscardState,

    pub(crate) scope: UsageScope<'scope>,

    pub(crate) binder: Binder,

    pub(crate) temp_offsets: Vec<u32>,

    pub(crate) dynamic_offset_count: usize,

    pub(crate) snatch_guard: &'snatch_guard SnatchGuard<'snatch_guard>,

    pub(crate) debug_scope_depth: u32,
    pub(crate) string_offset: usize,
}

pub(crate) fn set_bind_group<E>(
    state: &mut BaseState,
    cmd_enc: &CommandEncoder,
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

    let max_bind_groups = state.device.limits.max_bind_groups;
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

    if bind_group.is_none() {
        // TODO: Handle bind_group None.
        return Ok(());
    }

    let bind_group = bind_group.unwrap();
    let bind_group = state.tracker.bind_groups.insert_single(bind_group);

    bind_group.same_device_as(cmd_enc)?;

    bind_group.validate_dynamic_bindings(index, &state.temp_offsets)?;

    if merge_bind_groups {
        // merge the resource tracker in
        unsafe {
            state.scope.merge_bind_group(&bind_group.used)?;
        }
    }
    //Note: stateless trackers are not merged: the lifetime reference
    // is held to the bind group itself.

    state
        .buffer_memory_init_actions
        .extend(bind_group.used_buffer_ranges.iter().filter_map(|action| {
            action
                .buffer
                .initialization_status
                .read()
                .check_action(action)
        }));
    for action in bind_group.used_texture_ranges.iter() {
        state
            .pending_discard_init_fixups
            .extend(state.texture_memory_actions.register_init_action(action));
    }

    let used_resource = bind_group
        .used
        .acceleration_structures
        .into_iter()
        .map(|tlas| AsAction::UseTlas(tlas.clone()));

    state.as_actions.extend(used_resource);

    let pipeline_layout = state.binder.pipeline_layout.clone();
    let entries = state
        .binder
        .assign_group(index as usize, bind_group, &state.temp_offsets);
    if !entries.is_empty() && pipeline_layout.is_some() {
        let pipeline_layout = pipeline_layout.as_ref().unwrap().raw();
        for (i, e) in entries.iter().enumerate() {
            if let Some(group) = e.group.as_ref() {
                let raw_bg = group.try_raw(state.snatch_guard)?;
                unsafe {
                    state.raw_encoder.set_bind_group(
                        pipeline_layout,
                        index + i as u32,
                        Some(raw_bg),
                        &e.dynamic_offsets,
                    );
                }
            }
        }
    }
    Ok(())
}

/// After a pipeline has been changed, resources must be rebound
pub(crate) fn rebind_resources<E, F: FnOnce()>(
    state: &mut BaseState,
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
        let (start_index, entries) = state
            .binder
            .change_pipeline_layout(pipeline_layout, late_sized_buffer_groups);
        if !entries.is_empty() {
            for (i, e) in entries.iter().enumerate() {
                if let Some(group) = e.group.as_ref() {
                    let raw_bg = group.try_raw(state.snatch_guard)?;
                    unsafe {
                        state.raw_encoder.set_bind_group(
                            pipeline_layout.raw(),
                            start_index as u32 + i as u32,
                            Some(raw_bg),
                            &e.dynamic_offsets,
                        );
                    }
                }
            }
        }

        f();

        let non_overlapping =
            super::bind::compute_nonoverlapping_ranges(&pipeline_layout.push_constant_ranges);

        // Clear push constant ranges
        for range in non_overlapping {
            let offset = range.range.start;
            let size_bytes = range.range.end - offset;
            super::push_constant_clear(offset, size_bytes, |clear_offset, clear_data| unsafe {
                state.raw_encoder.set_push_constants(
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
    state: &mut BaseState,
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
            .raw_encoder
            .set_push_constants(pipeline_layout.raw(), stages, offset, data_slice)
    }
    Ok(())
}

pub(crate) fn write_timestamp<E>(
    state: &mut BaseState,
    cmd_enc: &CommandEncoder,
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

    query_set.same_device_as(cmd_enc)?;

    state
        .device
        .require_features(wgt::Features::TIMESTAMP_QUERY_INSIDE_PASSES)?;

    let query_set = state.tracker.query_sets.insert_single(query_set);

    query_set.validate_and_write_timestamp(state.raw_encoder, query_index, pending_query_resets)?;
    Ok(())
}

pub(crate) fn push_debug_group(state: &mut BaseState, string_data: &[u8], len: usize) {
    state.debug_scope_depth += 1;
    if !state
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        let label =
            str::from_utf8(&string_data[state.string_offset..state.string_offset + len]).unwrap();

        api_log!("Pass::push_debug_group {label:?}");
        unsafe {
            state.raw_encoder.begin_debug_marker(label);
        }
    }
    state.string_offset += len;
}

pub(crate) fn pop_debug_group<E>(state: &mut BaseState) -> Result<(), E>
where
    E: From<DebugGroupError>,
{
    api_log!("Pass::pop_debug_group");

    if state.debug_scope_depth == 0 {
        return Err(DebugGroupError::InvalidPop.into());
    }
    state.debug_scope_depth -= 1;
    if !state
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        unsafe {
            state.raw_encoder.end_debug_marker();
        }
    }
    Ok(())
}

pub(crate) fn insert_debug_marker(state: &mut BaseState, string_data: &[u8], len: usize) {
    if !state
        .device
        .instance_flags
        .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
    {
        let label =
            str::from_utf8(&string_data[state.string_offset..state.string_offset + len]).unwrap();
        api_log!("Pass::insert_debug_marker {label:?}");
        unsafe {
            state.raw_encoder.insert_debug_marker(label);
        }
    }
    state.string_offset += len;
}

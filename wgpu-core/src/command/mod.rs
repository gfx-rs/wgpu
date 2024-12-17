mod allocator;
mod bind;
mod bundle;
mod clear;
mod compute;
mod compute_command;
mod draw;
mod memory_init;
mod query;
mod ray_tracing;
mod render;
mod render_command;
mod timestamp_writes;
mod transfer;

use std::mem::{self, ManuallyDrop};
use std::sync::Arc;

pub(crate) use self::clear::clear_texture;
pub use self::{
    bundle::*, clear::ClearError, compute::*, compute_command::ComputeCommand, draw::*, query::*,
    render::*, render_command::RenderCommand, transfer::*,
};
pub(crate) use allocator::CommandAllocator;

pub(crate) use timestamp_writes::ArcPassTimestampWrites;
pub use timestamp_writes::PassTimestampWrites;

use self::memory_init::CommandBufferTextureMemoryActions;

use crate::device::{Device, DeviceError, MissingFeatures};
use crate::lock::{rank, Mutex};
use crate::snatch::SnatchGuard;

use crate::init_tracker::BufferInitTrackerAction;
use crate::ray_tracing::{BlasAction, TlasAction};
use crate::resource::{Fallible, InvalidResourceError, Labeled, ParentDevice as _, QuerySet};
use crate::storage::Storage;
use crate::track::{DeviceTracker, Tracker, UsageScope};
use crate::LabelHelpers;
use crate::{api_log, global::Global, id, resource_log, Label};

use thiserror::Error;

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;

const PUSH_CONSTANT_CLEAR_ARRAY: &[u32] = &[0_u32; 64];

/// The current state of a [`CommandBuffer`].
pub(crate) enum CommandEncoderStatus {
    /// Ready to record commands. An encoder's initial state.
    ///
    /// Command building methods like [`command_encoder_clear_buffer`] and
    /// [`compute_pass_end`] require the encoder to be in this
    /// state.
    ///
    /// This corresponds to WebGPU's "open" state.
    /// See <https://www.w3.org/TR/webgpu/#encoder-state-open>
    ///
    /// [`command_encoder_clear_buffer`]: Global::command_encoder_clear_buffer
    /// [`compute_pass_end`]: Global::compute_pass_end
    Recording(CommandBufferMutable),

    /// Locked by a render or compute pass.
    ///
    /// This state is entered when a render/compute pass is created,
    /// and exited when the pass is ended.
    ///
    /// As long as the command encoder is locked, any command building operation on it will fail
    /// and put the encoder into the [`Self::Error`] state.
    /// See <https://www.w3.org/TR/webgpu/#encoder-state-locked>
    Locked(CommandBufferMutable),

    /// Command recording is complete, and the buffer is ready for submission.
    ///
    /// [`Global::command_encoder_finish`] transitions a
    /// `CommandBuffer` from the `Recording` state into this state.
    ///
    /// [`Global::queue_submit`] drops command buffers unless they are
    /// in this state.
    Finished(CommandBufferMutable),

    /// An error occurred while recording a compute or render pass.
    ///
    /// When a `CommandEncoder` is left in this state, we have also
    /// returned an error result from the function that encountered
    /// the problem. Future attempts to use the encoder (for example,
    /// calls to [`Self::record`]) will also return errors.
    Error,
}

impl CommandEncoderStatus {
    /// Checks that the encoder is in the [`Self::Recording`] state.
    pub(crate) fn record(&mut self) -> Result<RecordingGuard<'_>, CommandEncoderError> {
        match self {
            Self::Recording(_) => Ok(RecordingGuard { inner: self }),
            Self::Locked(_) => {
                *self = Self::Error;
                Err(CommandEncoderError::Locked)
            }
            Self::Finished(_) => Err(CommandEncoderError::NotRecording),
            Self::Error => Err(CommandEncoderError::Invalid),
        }
    }

    #[cfg(feature = "trace")]
    fn get_inner(&mut self) -> Result<&mut CommandBufferMutable, CommandEncoderError> {
        match self {
            Self::Locked(inner) | Self::Finished(inner) | Self::Recording(inner) => Ok(inner),
            Self::Error => Err(CommandEncoderError::Invalid),
        }
    }

    /// Locks the encoder by putting it in the [`Self::Locked`] state.
    ///
    /// Call [`Self::unlock_encoder`] to put the [`CommandBuffer`] back into the [`Self::Recording`] state.
    fn lock_encoder(&mut self) -> Result<(), CommandEncoderError> {
        match mem::replace(self, Self::Error) {
            Self::Recording(inner) => {
                *self = Self::Locked(inner);
                Ok(())
            }
            Self::Finished(inner) => {
                *self = Self::Finished(inner);
                Err(CommandEncoderError::NotRecording)
            }
            Self::Locked(_) => Err(CommandEncoderError::Locked),
            Self::Error => Err(CommandEncoderError::Invalid),
        }
    }

    /// Unlocks the [`CommandBuffer`] and puts it back into the [`Self::Recording`] state.
    ///
    /// This function is the unlocking counterpart to [`Self::lock_encoder`].
    ///
    /// It is only valid to call this function if the encoder is in the [`Self::Locked`] state.
    fn unlock_encoder(&mut self) -> Result<RecordingGuard<'_>, CommandEncoderError> {
        match mem::replace(self, Self::Error) {
            Self::Locked(inner) => {
                *self = Self::Recording(inner);
                Ok(RecordingGuard { inner: self })
            }
            Self::Finished(inner) => {
                *self = Self::Finished(inner);
                Err(CommandEncoderError::NotRecording)
            }
            Self::Recording(_) => Err(CommandEncoderError::Invalid),
            Self::Error => Err(CommandEncoderError::Invalid),
        }
    }

    fn finish(&mut self, device: &Device) -> Result<(), CommandEncoderError> {
        match mem::replace(self, Self::Error) {
            Self::Recording(mut inner) => {
                if let Err(e) = inner.encoder.close(device) {
                    Err(e.into())
                } else {
                    *self = Self::Finished(inner);
                    // Note: if we want to stop tracking the swapchain texture view,
                    // this is the place to do it.
                    Ok(())
                }
            }
            Self::Finished(inner) => {
                *self = Self::Finished(inner);
                Err(CommandEncoderError::NotRecording)
            }
            Self::Locked(_) => Err(CommandEncoderError::Locked),
            Self::Error => Err(CommandEncoderError::Invalid),
        }
    }
}

/// A guard to enforce error reporting, for a [`CommandBuffer`] in the [`Recording`] state.
///
/// An [`RecordingGuard`] holds a mutable reference to a [`CommandEncoderStatus`] that
/// has been verified to be in the [`Recording`] state. The [`RecordingGuard`] dereferences
/// mutably to the [`CommandBufferMutable`] that the status holds.
///
/// Dropping an [`RecordingGuard`] sets the [`CommandBuffer`]'s state to
/// [`CommandEncoderStatus::Error`]. If your use of the guard was
/// successful, call its [`mark_successful`] method to dispose of it.
///
/// [`Recording`]: CommandEncoderStatus::Recording
/// [`mark_successful`]: Self::mark_successful
pub(crate) struct RecordingGuard<'a> {
    inner: &'a mut CommandEncoderStatus,
}

impl<'a> RecordingGuard<'a> {
    pub(crate) fn mark_successful(self) {
        mem::forget(self)
    }
}

impl<'a> Drop for RecordingGuard<'a> {
    fn drop(&mut self) {
        *self.inner = CommandEncoderStatus::Error;
    }
}

impl<'a> std::ops::Deref for RecordingGuard<'a> {
    type Target = CommandBufferMutable;

    fn deref(&self) -> &Self::Target {
        match &*self.inner {
            CommandEncoderStatus::Recording(command_buffer_mutable) => command_buffer_mutable,
            _ => unreachable!(),
        }
    }
}

impl<'a> std::ops::DerefMut for RecordingGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.inner {
            CommandEncoderStatus::Recording(command_buffer_mutable) => command_buffer_mutable,
            _ => unreachable!(),
        }
    }
}

/// A raw [`CommandEncoder`][rce], and the raw [`CommandBuffer`][rcb]s built from it.
///
/// Each wgpu-core [`CommandBuffer`] owns an instance of this type, which is
/// where the commands are actually stored.
///
/// This holds a `Vec` of raw [`CommandBuffer`][rcb]s, not just one. We are not
/// always able to record commands in the order in which they must ultimately be
/// submitted to the queue, but raw command buffers don't permit inserting new
/// commands into the middle of a recorded stream. However, hal queue submission
/// accepts a series of command buffers at once, so we can simply break the
/// stream up into multiple buffers, and then reorder the buffers. See
/// [`CommandEncoder::close_and_swap`] for a specific example of this.
///
/// Note that a [`CommandEncoderId`] actually refers to a [`CommandBuffer`].
/// Methods that take a command encoder id actually look up the command buffer,
/// and then use its encoder.
///
/// [rce]: hal::Api::CommandEncoder
/// [rcb]: hal::Api::CommandBuffer
/// [`CommandEncoderId`]: crate::id::CommandEncoderId
pub(crate) struct CommandEncoder {
    /// The underlying `wgpu_hal` [`CommandEncoder`].
    ///
    /// Successfully executed command buffers' encoders are saved in a
    /// [`CommandAllocator`] for recycling.
    ///
    /// [`CommandEncoder`]: hal::Api::CommandEncoder
    /// [`CommandAllocator`]: crate::command::CommandAllocator
    pub(crate) raw: ManuallyDrop<Box<dyn hal::DynCommandEncoder>>,

    /// All the raw command buffers for our owning [`CommandBuffer`], in
    /// submission order.
    ///
    /// These command buffers were all constructed with `raw`. The
    /// [`wgpu_hal::CommandEncoder`] trait forbids these from outliving `raw`,
    /// and requires that we provide all of these when we call
    /// [`raw.reset_all()`][CE::ra], so the encoder and its buffers travel
    /// together.
    ///
    /// [CE::ra]: hal::CommandEncoder::reset_all
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) list: Vec<Box<dyn hal::DynCommandBuffer>>,

    pub(crate) device: Arc<Device>,

    /// True if `raw` is in the "recording" state.
    ///
    /// See the documentation for [`wgpu_hal::CommandEncoder`] for
    /// details on the states `raw` can be in.
    ///
    /// [`wgpu_hal::CommandEncoder`]: hal::CommandEncoder
    pub(crate) is_open: bool,

    pub(crate) hal_label: Option<String>,
}

//TODO: handle errors better
impl CommandEncoder {
    /// Finish the current command buffer, if any, and place it
    /// at the second-to-last position in our list.
    ///
    /// If we have opened this command encoder, finish its current
    /// command buffer, and insert it just before the last element in
    /// [`self.list`][l]. If this command buffer is closed, do nothing.
    ///
    /// On return, the underlying hal encoder is closed.
    ///
    /// What is this for?
    ///
    /// The `wgpu_hal` contract requires that each render or compute pass's
    /// commands be preceded by calls to [`transition_buffers`] and
    /// [`transition_textures`], to put the resources the pass operates on in
    /// the appropriate state. Unfortunately, we don't know which transitions
    /// are needed until we're done recording the pass itself. Rather than
    /// iterating over the pass twice, we note the necessary transitions as we
    /// record its commands, finish the raw command buffer for the actual pass,
    /// record a new raw command buffer for the transitions, and jam that buffer
    /// in just before the pass's. This is the function that jams in the
    /// transitions' command buffer.
    ///
    /// [l]: CommandEncoder::list
    /// [`transition_buffers`]: hal::CommandEncoder::transition_buffers
    /// [`transition_textures`]: hal::CommandEncoder::transition_textures
    fn close_and_swap(&mut self, device: &Device) -> Result<(), DeviceError> {
        if self.is_open {
            self.is_open = false;
            let new = unsafe { self.raw.end_encoding() }.map_err(|e| device.handle_hal_error(e))?;
            self.list.insert(self.list.len() - 1, new);
        }

        Ok(())
    }

    /// Finish the current command buffer, if any, and add it to the
    /// end of [`self.list`][l].
    ///
    /// If we have opened this command encoder, finish its current
    /// command buffer, and push it onto the end of [`self.list`][l].
    /// If this command buffer is closed, do nothing.
    ///
    /// On return, the underlying hal encoder is closed.
    ///
    /// [l]: CommandEncoder::list
    fn close(&mut self, device: &Device) -> Result<(), DeviceError> {
        if self.is_open {
            self.is_open = false;
            let cmd_buf =
                unsafe { self.raw.end_encoding() }.map_err(|e| device.handle_hal_error(e))?;
            self.list.push(cmd_buf);
        }

        Ok(())
    }

    /// Begin recording a new command buffer, if we haven't already.
    ///
    /// The underlying hal encoder is put in the "recording" state.
    pub(crate) fn open(
        &mut self,
        device: &Device,
    ) -> Result<&mut dyn hal::DynCommandEncoder, DeviceError> {
        if !self.is_open {
            self.is_open = true;
            let hal_label = self.hal_label.as_deref();
            unsafe { self.raw.begin_encoding(hal_label) }
                .map_err(|e| device.handle_hal_error(e))?;
        }

        Ok(self.raw.as_mut())
    }

    /// Begin recording a new command buffer for a render pass, with
    /// its own label.
    ///
    /// The underlying hal encoder is put in the "recording" state.
    fn open_pass(&mut self, hal_label: Option<&str>, device: &Device) -> Result<(), DeviceError> {
        self.is_open = true;
        unsafe { self.raw.begin_encoding(hal_label) }.map_err(|e| device.handle_hal_error(e))?;

        Ok(())
    }
}

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if self.is_open {
            unsafe { self.raw.discard_encoding() };
        }
        unsafe {
            self.raw.reset_all(mem::take(&mut self.list));
        }
        // SAFETY: We are in the Drop impl and we don't use self.raw anymore after this point.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        self.device.command_allocator.release_encoder(raw);
    }
}

/// Look at the documentation for [`CommandBufferMutable`] for an explanation of
/// the fields in this struct. This is the "built" counterpart to that type.
pub(crate) struct BakedCommands {
    pub(crate) encoder: CommandEncoder,
    pub(crate) trackers: Tracker,
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_actions: CommandBufferTextureMemoryActions,
}

/// The mutable state of a [`CommandBuffer`].
pub struct CommandBufferMutable {
    /// The [`wgpu_hal::Api::CommandBuffer`]s we've built so far, and the encoder
    /// they belong to.
    ///
    /// [`wgpu_hal::Api::CommandBuffer`]: hal::Api::CommandBuffer
    pub(crate) encoder: CommandEncoder,

    /// All the resources that the commands recorded so far have referred to.
    pub(crate) trackers: Tracker,

    /// The regions of buffers and textures these commands will read and write.
    ///
    /// This is used to determine which portions of which
    /// buffers/textures we actually need to initialize. If we're
    /// definitely going to write to something before we read from it,
    /// we don't need to clear its contents.
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_actions: CommandBufferTextureMemoryActions,

    pub(crate) pending_query_resets: QueryResetMap,

    blas_actions: Vec<BlasAction>,
    tlas_actions: Vec<TlasAction>,

    #[cfg(feature = "trace")]
    pub(crate) commands: Option<Vec<TraceCommand>>,
}

impl CommandBufferMutable {
    pub(crate) fn open_encoder_and_tracker(
        &mut self,
        device: &Device,
    ) -> Result<(&mut dyn hal::DynCommandEncoder, &mut Tracker), DeviceError> {
        let encoder = self.encoder.open(device)?;
        let tracker = &mut self.trackers;

        Ok((encoder, tracker))
    }

    pub(crate) fn into_baked_commands(self) -> BakedCommands {
        BakedCommands {
            encoder: self.encoder,
            trackers: self.trackers,
            buffer_memory_init_actions: self.buffer_memory_init_actions,
            texture_memory_actions: self.texture_memory_actions,
        }
    }
}

/// A buffer of commands to be submitted to the GPU for execution.
///
/// Whereas the WebGPU API uses two separate types for command buffers and
/// encoders, this type is a fusion of the two:
///
/// - During command recording, this holds a [`CommandEncoder`] accepting this
///   buffer's commands. In this state, the [`CommandBuffer`] type behaves like
///   a WebGPU `GPUCommandEncoder`.
///
/// - Once command recording is finished by calling
///   [`Global::command_encoder_finish`], no further recording is allowed. The
///   internal [`CommandEncoder`] is retained solely as a storage pool for the
///   raw command buffers. In this state, the value behaves like a WebGPU
///   `GPUCommandBuffer`.
///
/// - Once a command buffer is submitted to the queue, it is removed from the id
///   registry, and its contents are taken to construct a [`BakedCommands`],
///   whose contents eventually become the property of the submission queue.
pub struct CommandBuffer {
    pub(crate) device: Arc<Device>,
    support_clear_texture: bool,
    /// The `label` from the descriptor used to create the resource.
    label: String,

    /// The mutable state of this command buffer.
    pub(crate) data: Mutex<CommandEncoderStatus>,
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        resource_log!("Drop {}", self.error_ident());
    }
}

impl CommandBuffer {
    pub(crate) fn new(
        encoder: Box<dyn hal::DynCommandEncoder>,
        device: &Arc<Device>,
        label: &Label,
    ) -> Self {
        CommandBuffer {
            device: device.clone(),
            support_clear_texture: device.features.contains(wgt::Features::CLEAR_TEXTURE),
            label: label.to_string(),
            data: Mutex::new(
                rank::COMMAND_BUFFER_DATA,
                CommandEncoderStatus::Recording(CommandBufferMutable {
                    encoder: CommandEncoder {
                        raw: ManuallyDrop::new(encoder),
                        list: Vec::new(),
                        device: device.clone(),
                        is_open: false,
                        hal_label: label.to_hal(device.instance_flags).map(str::to_owned),
                    },
                    trackers: Tracker::new(),
                    buffer_memory_init_actions: Default::default(),
                    texture_memory_actions: Default::default(),
                    pending_query_resets: QueryResetMap::new(),
                    blas_actions: Default::default(),
                    tlas_actions: Default::default(),
                    #[cfg(feature = "trace")]
                    commands: if device.trace.lock().is_some() {
                        Some(Vec::new())
                    } else {
                        None
                    },
                }),
            ),
        }
    }

    pub(crate) fn new_invalid(device: &Arc<Device>, label: &Label) -> Self {
        CommandBuffer {
            device: device.clone(),
            support_clear_texture: device.features.contains(wgt::Features::CLEAR_TEXTURE),
            label: label.to_string(),
            data: Mutex::new(rank::COMMAND_BUFFER_DATA, CommandEncoderStatus::Error),
        }
    }

    pub(crate) fn insert_barriers_from_tracker(
        raw: &mut dyn hal::DynCommandEncoder,
        base: &mut Tracker,
        head: &Tracker,
        snatch_guard: &SnatchGuard,
    ) {
        profiling::scope!("insert_barriers");

        base.buffers.set_from_tracker(&head.buffers);
        base.textures.set_from_tracker(&head.textures);

        Self::drain_barriers(raw, base, snatch_guard);
    }

    pub(crate) fn insert_barriers_from_scope(
        raw: &mut dyn hal::DynCommandEncoder,
        base: &mut Tracker,
        head: &UsageScope,
        snatch_guard: &SnatchGuard,
    ) {
        profiling::scope!("insert_barriers");

        base.buffers.set_from_usage_scope(&head.buffers);
        base.textures.set_from_usage_scope(&head.textures);

        Self::drain_barriers(raw, base, snatch_guard);
    }

    pub(crate) fn drain_barriers(
        raw: &mut dyn hal::DynCommandEncoder,
        base: &mut Tracker,
        snatch_guard: &SnatchGuard,
    ) {
        profiling::scope!("drain_barriers");

        let buffer_barriers = base
            .buffers
            .drain_transitions(snatch_guard)
            .collect::<Vec<_>>();
        let (transitions, textures) = base.textures.drain_transitions(snatch_guard);
        let texture_barriers = transitions
            .into_iter()
            .enumerate()
            .map(|(i, p)| p.into_hal(textures[i].unwrap().raw()))
            .collect::<Vec<_>>();

        unsafe {
            raw.transition_buffers(&buffer_barriers);
            raw.transition_textures(&texture_barriers);
        }
    }

    pub(crate) fn insert_barriers_from_device_tracker(
        raw: &mut dyn hal::DynCommandEncoder,
        base: &mut DeviceTracker,
        head: &Tracker,
        snatch_guard: &SnatchGuard,
    ) {
        profiling::scope!("insert_barriers_from_device_tracker");

        let buffer_barriers = base
            .buffers
            .set_from_tracker_and_drain_transitions(&head.buffers, snatch_guard)
            .collect::<Vec<_>>();

        let texture_barriers = base
            .textures
            .set_from_tracker_and_drain_transitions(&head.textures, snatch_guard)
            .collect::<Vec<_>>();

        unsafe {
            raw.transition_buffers(&buffer_barriers);
            raw.transition_textures(&texture_barriers);
        }
    }
}

impl CommandBuffer {
    pub fn take_finished<'a>(&'a self) -> Result<CommandBufferMutable, InvalidResourceError> {
        let status = mem::replace(&mut *self.data.lock(), CommandEncoderStatus::Error);
        match status {
            CommandEncoderStatus::Finished(command_buffer_mutable) => Ok(command_buffer_mutable),
            CommandEncoderStatus::Recording(_)
            | CommandEncoderStatus::Locked(_)
            | CommandEncoderStatus::Error => Err(InvalidResourceError(self.error_ident())),
        }
    }
}

crate::impl_resource_type!(CommandBuffer);
crate::impl_labeled!(CommandBuffer);
crate::impl_parent_device!(CommandBuffer);
crate::impl_storage_item!(CommandBuffer);

/// A stream of commands for a render pass or compute pass.
///
/// This also contains side tables referred to by certain commands,
/// like dynamic offsets for [`SetBindGroup`] or string data for
/// [`InsertDebugMarker`].
///
/// Render passes use `BasePass<RenderCommand>`, whereas compute
/// passes use `BasePass<ComputeCommand>`.
///
/// [`SetBindGroup`]: RenderCommand::SetBindGroup
/// [`InsertDebugMarker`]: RenderCommand::InsertDebugMarker
#[doc(hidden)]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BasePass<C> {
    pub label: Option<String>,

    /// The stream of commands.
    pub commands: Vec<C>,

    /// Dynamic offsets consumed by [`SetBindGroup`] commands in `commands`.
    ///
    /// Each successive `SetBindGroup` consumes the next
    /// [`num_dynamic_offsets`] values from this list.
    pub dynamic_offsets: Vec<wgt::DynamicOffset>,

    /// Strings used by debug instructions.
    ///
    /// Each successive [`PushDebugGroup`] or [`InsertDebugMarker`]
    /// instruction consumes the next `len` bytes from this vector.
    pub string_data: Vec<u8>,

    /// Data used by `SetPushConstant` instructions.
    ///
    /// See the documentation for [`RenderCommand::SetPushConstant`]
    /// and [`ComputeCommand::SetPushConstant`] for details.
    pub push_constant_data: Vec<u32>,
}

impl<C: Clone> BasePass<C> {
    fn new(label: &Label) -> Self {
        Self {
            label: label.as_ref().map(|cow| cow.to_string()),
            commands: Vec::new(),
            dynamic_offsets: Vec::new(),
            string_data: Vec::new(),
            push_constant_data: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CommandEncoderError {
    #[error("Command encoder is invalid")]
    Invalid,
    #[error("Command encoder must be active")]
    NotRecording,
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error("Command encoder is locked by a previously created render/compute pass. Before recording any new commands, the pass must be ended.")]
    Locked,

    #[error(transparent)]
    InvalidColorAttachment(#[from] ColorAttachmentError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(
        "begin and end indices of pass timestamp writes are both set to {idx}, which is not allowed"
    )]
    TimestampWriteIndicesEqual { idx: u32 },
    #[error(transparent)]
    TimestampWritesInvalid(#[from] QueryUseError),
    #[error("no begin or end indices were specified for pass timestamp writes, expected at least one to be set")]
    TimestampWriteIndicesMissing,
}

impl Global {
    pub fn command_encoder_finish(
        &self,
        encoder_id: id::CommandEncoderId,
        _desc: &wgt::CommandBufferDescriptor<Label>,
    ) -> (id::CommandBufferId, Option<CommandEncoderError>) {
        profiling::scope!("CommandEncoder::finish");

        let hub = &self.hub;

        let cmd_buf = hub.command_buffers.get(encoder_id.into_command_buffer_id());

        let error = match cmd_buf.data.lock().finish(&cmd_buf.device) {
            Ok(_) => None,
            Err(e) => Some(e),
        };

        (encoder_id.into_command_buffer_id(), error)
    }

    pub fn command_encoder_push_debug_group(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::push_debug_group");
        api_log!("CommandEncoder::push_debug_group {label}");

        let hub = &self.hub;

        let cmd_buf = hub.command_buffers.get(encoder_id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.data.lock();
        let mut cmd_buf_data_guard = cmd_buf_data.record()?;
        let cmd_buf_data = &mut *cmd_buf_data_guard;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::PushDebugGroup(label.to_string()));
        }

        let cmd_buf_raw = cmd_buf_data.encoder.open(&cmd_buf.device)?;
        if !cmd_buf
            .device
            .instance_flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
        {
            unsafe {
                cmd_buf_raw.begin_debug_marker(label);
            }
        }

        cmd_buf_data_guard.mark_successful();
        Ok(())
    }

    pub fn command_encoder_insert_debug_marker(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::insert_debug_marker");
        api_log!("CommandEncoder::insert_debug_marker {label}");

        let hub = &self.hub;

        let cmd_buf = hub.command_buffers.get(encoder_id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.data.lock();
        let mut cmd_buf_data_guard = cmd_buf_data.record()?;
        let cmd_buf_data = &mut *cmd_buf_data_guard;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::InsertDebugMarker(label.to_string()));
        }

        if !cmd_buf
            .device
            .instance_flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
        {
            let cmd_buf_raw = cmd_buf_data.encoder.open(&cmd_buf.device)?;
            unsafe {
                cmd_buf_raw.insert_debug_marker(label);
            }
        }

        cmd_buf_data_guard.mark_successful();
        Ok(())
    }

    pub fn command_encoder_pop_debug_group(
        &self,
        encoder_id: id::CommandEncoderId,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::pop_debug_marker");
        api_log!("CommandEncoder::pop_debug_group");

        let hub = &self.hub;

        let cmd_buf = hub.command_buffers.get(encoder_id.into_command_buffer_id());
        let mut cmd_buf_data = cmd_buf.data.lock();
        let mut cmd_buf_data_guard = cmd_buf_data.record()?;
        let cmd_buf_data = &mut *cmd_buf_data_guard;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf_data.commands {
            list.push(TraceCommand::PopDebugGroup);
        }

        let cmd_buf_raw = cmd_buf_data.encoder.open(&cmd_buf.device)?;
        if !cmd_buf
            .device
            .instance_flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
        {
            unsafe {
                cmd_buf_raw.end_debug_marker();
            }
        }

        cmd_buf_data_guard.mark_successful();
        Ok(())
    }

    fn validate_pass_timestamp_writes(
        device: &Device,
        query_sets: &Storage<Fallible<QuerySet>>,
        timestamp_writes: &PassTimestampWrites,
    ) -> Result<ArcPassTimestampWrites, CommandEncoderError> {
        let &PassTimestampWrites {
            query_set,
            beginning_of_pass_write_index,
            end_of_pass_write_index,
        } = timestamp_writes;

        device.require_features(wgt::Features::TIMESTAMP_QUERY)?;

        let query_set = query_sets.get(query_set).get()?;

        query_set.same_device(device)?;

        for idx in [beginning_of_pass_write_index, end_of_pass_write_index]
            .into_iter()
            .flatten()
        {
            query_set.validate_query(SimplifiedQueryType::Timestamp, idx, None)?;
        }

        if let Some((begin, end)) = beginning_of_pass_write_index.zip(end_of_pass_write_index) {
            if begin == end {
                return Err(CommandEncoderError::TimestampWriteIndicesEqual { idx: begin });
            }
        }

        if beginning_of_pass_write_index
            .or(end_of_pass_write_index)
            .is_none()
        {
            return Err(CommandEncoderError::TimestampWriteIndicesMissing);
        }

        Ok(ArcPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index,
            end_of_pass_write_index,
        })
    }
}

fn push_constant_clear<PushFn>(offset: u32, size_bytes: u32, mut push_fn: PushFn)
where
    PushFn: FnMut(u32, &[u32]),
{
    let mut count_words = 0_u32;
    let size_words = size_bytes / wgt::PUSH_CONSTANT_ALIGNMENT;
    while count_words < size_words {
        let count_bytes = count_words * wgt::PUSH_CONSTANT_ALIGNMENT;
        let size_to_write_words =
            (size_words - count_words).min(PUSH_CONSTANT_CLEAR_ARRAY.len() as u32);

        push_fn(
            offset + count_bytes,
            &PUSH_CONSTANT_CLEAR_ARRAY[0..size_to_write_words as usize],
        );

        count_words += size_to_write_words;
    }
}

#[derive(Debug, Copy, Clone)]
struct StateChange<T> {
    last_state: Option<T>,
}

impl<T: Copy + PartialEq> StateChange<T> {
    fn new() -> Self {
        Self { last_state: None }
    }
    fn set_and_check_redundant(&mut self, new_state: T) -> bool {
        let already_set = self.last_state == Some(new_state);
        self.last_state = Some(new_state);
        already_set
    }
    fn reset(&mut self) {
        self.last_state = None;
    }
}

impl<T: Copy + PartialEq> Default for StateChange<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct BindGroupStateChange {
    last_states: [StateChange<Option<id::BindGroupId>>; hal::MAX_BIND_GROUPS],
}

impl BindGroupStateChange {
    fn new() -> Self {
        Self {
            last_states: [StateChange::new(); hal::MAX_BIND_GROUPS],
        }
    }

    fn set_and_check_redundant(
        &mut self,
        bind_group_id: Option<id::BindGroupId>,
        index: u32,
        dynamic_offsets: &mut Vec<u32>,
        offsets: &[wgt::DynamicOffset],
    ) -> bool {
        // For now never deduplicate bind groups with dynamic offsets.
        if offsets.is_empty() {
            // If this get returns None, that means we're well over the limit,
            // so let the call through to get a proper error
            if let Some(current_bind_group) = self.last_states.get_mut(index as usize) {
                // Bail out if we're binding the same bind group.
                if current_bind_group.set_and_check_redundant(bind_group_id) {
                    return true;
                }
            }
        } else {
            // We intentionally remove the memory of this bind group if we have dynamic offsets,
            // such that if you try to bind this bind group later with _no_ dynamic offsets it
            // tries to bind it again and gives a proper validation error.
            if let Some(current_bind_group) = self.last_states.get_mut(index as usize) {
                current_bind_group.reset();
            }
            dynamic_offsets.extend_from_slice(offsets);
        }
        false
    }
    fn reset(&mut self) {
        self.last_states = [StateChange::new(); hal::MAX_BIND_GROUPS];
    }
}

impl Default for BindGroupStateChange {
    fn default() -> Self {
        Self::new()
    }
}

trait MapPassErr<T, O> {
    fn map_pass_err(self, scope: PassErrorScope) -> Result<T, O>;
}

#[derive(Clone, Copy, Debug)]
pub enum DrawKind {
    Draw,
    DrawIndirect,
    MultiDrawIndirect,
    MultiDrawIndirectCount,
}

#[derive(Clone, Copy, Debug, Error)]
pub enum PassErrorScope {
    // TODO: Extract out the 2 error variants below so that we can always
    // include the ResourceErrorIdent of the pass around all inner errors
    #[error("In a bundle parameter")]
    Bundle,
    #[error("In a pass parameter")]
    Pass,
    #[error("In a set_bind_group command")]
    SetBindGroup,
    #[error("In a set_pipeline command")]
    SetPipelineRender,
    #[error("In a set_pipeline command")]
    SetPipelineCompute,
    #[error("In a set_push_constant command")]
    SetPushConstant,
    #[error("In a set_vertex_buffer command")]
    SetVertexBuffer,
    #[error("In a set_index_buffer command")]
    SetIndexBuffer,
    #[error("In a set_blend_constant command")]
    SetBlendConstant,
    #[error("In a set_stencil_reference command")]
    SetStencilReference,
    #[error("In a set_viewport command")]
    SetViewport,
    #[error("In a set_scissor_rect command")]
    SetScissorRect,
    #[error("In a draw command, kind: {kind:?}")]
    Draw { kind: DrawKind, indexed: bool },
    #[error("In a write_timestamp command")]
    WriteTimestamp,
    #[error("In a begin_occlusion_query command")]
    BeginOcclusionQuery,
    #[error("In a end_occlusion_query command")]
    EndOcclusionQuery,
    #[error("In a begin_pipeline_statistics_query command")]
    BeginPipelineStatisticsQuery,
    #[error("In a end_pipeline_statistics_query command")]
    EndPipelineStatisticsQuery,
    #[error("In a execute_bundle command")]
    ExecuteBundle,
    #[error("In a dispatch command, indirect:{indirect}")]
    Dispatch { indirect: bool },
    #[error("In a push_debug_group command")]
    PushDebugGroup,
    #[error("In a pop_debug_group command")]
    PopDebugGroup,
    #[error("In a insert_debug_marker command")]
    InsertDebugMarker,
}

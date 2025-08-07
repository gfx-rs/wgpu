mod allocator;
mod bind;
mod bundle;
mod clear;
mod compute;
mod compute_command;
mod draw;
mod memory_init;
mod pass;
mod query;
mod ray_tracing;
mod render;
mod render_command;
mod timestamp_writes;
mod transfer;
mod transition_resources;

use alloc::{borrow::ToOwned as _, boxed::Box, string::String, sync::Arc, vec::Vec};
use core::mem::{self, ManuallyDrop};
use core::ops;

pub(crate) use self::clear::clear_texture;
pub use self::{
    bundle::*, clear::ClearError, compute::*, compute_command::ComputeCommand, draw::*, query::*,
    render::*, render_command::RenderCommand, transfer::*,
};
pub(crate) use allocator::CommandAllocator;

pub(crate) use timestamp_writes::ArcPassTimestampWrites;
pub use timestamp_writes::PassTimestampWrites;

use self::memory_init::CommandBufferTextureMemoryActions;

use crate::binding_model::BindingError;
use crate::command::transition_resources::TransitionResourcesError;
use crate::device::queue::TempResource;
use crate::device::{Device, DeviceError, MissingFeatures};
use crate::lock::{rank, Mutex};
use crate::snatch::SnatchGuard;

use crate::init_tracker::BufferInitTrackerAction;
use crate::ray_tracing::{AsAction, BuildAccelerationStructureError};
use crate::resource::{
    DestroyedResourceError, Fallible, InvalidResourceError, Labeled, ParentDevice as _, QuerySet,
};
use crate::storage::Storage;
use crate::track::{DeviceTracker, ResourceUsageCompatibilityError, Tracker, UsageScope};
use crate::{api_log, global::Global, id, resource_log, Label};
use crate::{hal_label, LabelHelpers};

use wgt::error::{ErrorType, WebGpuError};

use thiserror::Error;

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;

const PUSH_CONSTANT_CLEAR_ARRAY: &[u32] = &[0_u32; 64];

/// The current state of a command or pass encoder.
///
/// In the WebGPU spec, the state of an encoder (open, locked, or ended) is
/// orthogonal to the validity of the encoder. However, this enum does not
/// represent the state of an invalid encoder.
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
    /// As long as the command encoder is locked, any command building operation
    /// on it will fail and put the encoder into the [`Self::Error`] state. See
    /// <https://www.w3.org/TR/webgpu/#encoder-state-locked>
    Locked(CommandBufferMutable),

    Consumed,

    /// Command recording is complete, and the buffer is ready for submission.
    ///
    /// [`Global::command_encoder_finish`] transitions a
    /// `CommandBuffer` from the `Recording` state into this state.
    ///
    /// [`Global::queue_submit`] requires that command buffers are
    /// in this state.
    ///
    /// This corresponds to WebGPU's "ended" state.
    /// See <https://www.w3.org/TR/webgpu/#encoder-state-ended>
    Finished(CommandBufferMutable),

    /// The command encoder is invalid.
    ///
    /// The error that caused the invalidation is stored here, and will
    /// be raised by `CommandEncoder.finish()`.
    Error(CommandEncoderError),

    /// Temporary state used internally by methods on `CommandEncoderStatus`.
    /// Encoder should never be left in this state.
    Transitioning,
}

impl CommandEncoderStatus {
    /// Record commands using the supplied closure.
    ///
    /// If the encoder is in the [`Self::Recording`] state, calls the closure to
    /// record commands. If the closure returns an error, stores that error in
    /// the encoder for later reporting when `finish()` is called. Returns
    /// `Ok(())` even if the closure returned an error.
    ///
    /// If the encoder is not in the [`Self::Recording`] state, the closure will
    /// not be called and nothing will be recorded. The encoder will be
    /// invalidated (if it is not already). If the error is a [validation error
    /// that should be raised immediately][ves], returns it in `Err`, otherwise,
    /// returns `Ok(())`.
    ///
    /// [ves]: https://www.w3.org/TR/webgpu/#abstract-opdef-validate-the-encoder-state
    fn record_with<
        F: FnOnce(&mut CommandBufferMutable) -> Result<(), E>,
        E: Clone + Into<CommandEncoderError>,
    >(
        &mut self,
        f: F,
    ) -> Result<(), EncoderStateError> {
        match self {
            Self::Recording(_) => {
                RecordingGuard { inner: self }.record(f);
                Ok(())
            }
            Self::Locked(_) => {
                // Invalidate the encoder and do not record anything, but do not
                // return an immediate validation error.
                self.invalidate(EncoderStateError::Locked);
                Ok(())
            }
            // Encoder is ended. Invalidate the encoder, do not record anything,
            // and return an immediate validation error.
            Self::Finished(_) => Err(self.invalidate(EncoderStateError::Ended)),
            Self::Consumed => Err(EncoderStateError::Ended),
            // Encoder is already invalid. Do not record anything, but do not
            // return an immediate validation error.
            Self::Error(_) => Ok(()),
            Self::Transitioning => unreachable!(),
        }
    }

    /// Special version of record used by `command_encoder_as_hal_mut`. This
    /// differs from the regular version in two ways:
    ///
    /// 1. The recording closure is infallible.
    /// 2. The recording closure takes `Option<&mut CommandBufferMutable>`, and
    ///    in the case that the encoder is not in a valid state for recording, the
    ///    closure is still called, with `None` as its argument.
    pub(crate) fn record_as_hal_mut<T, F: FnOnce(Option<&mut CommandBufferMutable>) -> T>(
        &mut self,
        f: F,
    ) -> T {
        match self {
            Self::Recording(_) => RecordingGuard { inner: self }.record_as_hal_mut(f),
            Self::Locked(_) => {
                self.invalidate(EncoderStateError::Locked);
                f(None)
            }
            Self::Finished(_) => {
                self.invalidate(EncoderStateError::Ended);
                f(None)
            }
            Self::Consumed => f(None),
            Self::Error(_) => f(None),
            Self::Transitioning => unreachable!(),
        }
    }

    #[cfg(feature = "trace")]
    fn get_inner(&mut self) -> &mut CommandBufferMutable {
        match self {
            Self::Locked(inner) | Self::Finished(inner) | Self::Recording(inner) => inner,
            // This is unreachable because this function is only used when
            // playing back a recorded trace. If only to avoid having to
            // implement serialization for all the error types, we don't support
            // storing the errors in a trace.
            Self::Consumed => unreachable!("command encoder is consumed"),
            Self::Error(_) => unreachable!("passes in a trace do not store errors"),
            Self::Transitioning => unreachable!(),
        }
    }

    /// Locks the encoder by putting it in the [`Self::Locked`] state.
    ///
    /// Render or compute passes call this on start. At the end of the pass,
    /// they call [`Self::unlock_and_record`] to put the [`CommandBuffer`] back
    /// into the [`Self::Recording`] state.
    fn lock_encoder(&mut self) -> Result<(), EncoderStateError> {
        match mem::replace(self, Self::Transitioning) {
            Self::Recording(inner) => {
                *self = Self::Locked(inner);
                Ok(())
            }
            st @ Self::Finished(_) => {
                // Attempting to open a pass on a finished encoder raises a
                // validation error but does not invalidate the encoder. This is
                // related to https://github.com/gpuweb/gpuweb/issues/5207.
                *self = st;
                Err(EncoderStateError::Ended)
            }
            Self::Locked(_) => Err(self.invalidate(EncoderStateError::Locked)),
            st @ Self::Consumed => {
                *self = st;
                Err(EncoderStateError::Ended)
            }
            st @ Self::Error(_) => {
                *self = st;
                Err(EncoderStateError::Invalid)
            }
            Self::Transitioning => unreachable!(),
        }
    }

    /// Unlocks the [`CommandBuffer`] and puts it back into the
    /// [`Self::Recording`] state, then records commands using the supplied
    /// closure.
    ///
    /// This function is the unlocking counterpart to [`Self::lock_encoder`]. It
    /// is only valid to call this function if the encoder is in the
    /// [`Self::Locked`] state.
    ///
    /// If the closure returns an error, stores that error in the encoder for
    /// later reporting when `finish()` is called. Returns `Ok(())` even if the
    /// closure returned an error.
    ///
    /// If the encoder is not in the [`Self::Locked`] state, the closure will
    /// not be called and nothing will be recorded. If a validation error should
    /// be raised immediately, returns it in `Err`, otherwise, returns `Ok(())`.
    fn unlock_and_record<
        F: FnOnce(&mut CommandBufferMutable) -> Result<(), E>,
        E: Clone + Into<CommandEncoderError>,
    >(
        &mut self,
        f: F,
    ) -> Result<(), EncoderStateError> {
        match mem::replace(self, Self::Transitioning) {
            Self::Locked(inner) => {
                *self = Self::Recording(inner);
                RecordingGuard { inner: self }.record(f);
                Ok(())
            }
            st @ Self::Finished(_) => {
                *self = st;
                Err(EncoderStateError::Ended)
            }
            Self::Recording(_) => {
                *self = Self::Error(EncoderStateError::Unlocked.into());
                Err(EncoderStateError::Unlocked)
            }
            st @ Self::Consumed => {
                *self = st;
                Err(EncoderStateError::Ended)
            }
            st @ Self::Error(_) => {
                // Encoder is invalid. Do not record anything, but do not
                // return an immediate validation error.
                *self = st;
                Ok(())
            }
            Self::Transitioning => unreachable!(),
        }
    }

    fn finish(&mut self) -> Self {
        // Replace our state with `Consumed`, and return either the inner
        // state or an error, to be transferred to the command buffer.
        match mem::replace(self, Self::Consumed) {
            Self::Recording(mut inner) => {
                if let Err(err) = inner.encoder.close_if_open() {
                    Self::Error(err.into())
                } else if inner.debug_scope_depth > 0 {
                    Self::Error(CommandEncoderError::DebugGroupError(
                        DebugGroupError::MissingPop,
                    ))
                } else {
                    // Note: if we want to stop tracking the swapchain texture view,
                    // this is the place to do it.
                    Self::Finished(inner)
                }
            }
            Self::Consumed | Self::Finished(_) => Self::Error(EncoderStateError::Ended.into()),
            Self::Locked(_) => Self::Error(EncoderStateError::Locked.into()),
            st @ Self::Error(_) => st,
            Self::Transitioning => unreachable!(),
        }
    }

    // Invalidate the command encoder and store the error `err` causing the
    // invalidation for diagnostic purposes.
    //
    // Since we do not track the state of an invalid encoder, it is not
    // necessary to unlock an encoder that has been invalidated.
    fn invalidate<E: Clone + Into<CommandEncoderError>>(&mut self, err: E) -> E {
        let enc_err = err.clone().into();
        api_log!("Invalidating command encoder: {enc_err:?}");
        *self = Self::Error(enc_err);
        err
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

    fn record<
        F: FnOnce(&mut CommandBufferMutable) -> Result<(), E>,
        E: Clone + Into<CommandEncoderError>,
    >(
        mut self,
        f: F,
    ) {
        match f(&mut self) {
            Ok(()) => self.mark_successful(),
            Err(err) => {
                self.inner.invalidate(err);
            }
        }
    }

    /// Special version of record used by `command_encoder_as_hal_mut`. This
    /// version takes an infallible recording closure.
    pub(crate) fn record_as_hal_mut<T, F: FnOnce(Option<&mut CommandBufferMutable>) -> T>(
        mut self,
        f: F,
    ) -> T {
        let res = f(Some(&mut self));
        self.mark_successful();
        res
    }
}

impl<'a> Drop for RecordingGuard<'a> {
    fn drop(&mut self) {
        if matches!(*self.inner, CommandEncoderStatus::Error(_)) {
            // Don't overwrite an error that is already present.
            return;
        }
        self.inner.invalidate(EncoderStateError::Invalid);
    }
}

impl<'a> ops::Deref for RecordingGuard<'a> {
    type Target = CommandBufferMutable;

    fn deref(&self) -> &Self::Target {
        match &*self.inner {
            CommandEncoderStatus::Recording(command_buffer_mutable) => command_buffer_mutable,
            _ => unreachable!(),
        }
    }
}

impl<'a> ops::DerefMut for RecordingGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self.inner {
            CommandEncoderStatus::Recording(command_buffer_mutable) => command_buffer_mutable,
            _ => unreachable!(),
        }
    }
}

pub(crate) struct CommandEncoder {
    pub(crate) device: Arc<Device>,

    pub(crate) label: String,

    /// The mutable state of this command encoder.
    pub(crate) data: Mutex<CommandEncoderStatus>,
}

crate::impl_resource_type!(CommandEncoder);
crate::impl_labeled!(CommandEncoder);
crate::impl_parent_device!(CommandEncoder);
crate::impl_storage_item!(CommandEncoder);

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        resource_log!("Drop {}", self.error_ident());
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
/// [`InnerCommandEncoder::close_and_swap`] for a specific example of this.
///
/// [rce]: hal::Api::CommandEncoder
/// [rcb]: hal::Api::CommandBuffer
pub(crate) struct InnerCommandEncoder {
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

    pub(crate) label: String,
}

impl InnerCommandEncoder {
    /// Finish the current command buffer and insert it just before
    /// the last element in [`self.list`][l].
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
    /// # Panics
    ///
    /// - If the encoder is not open.
    ///
    /// [l]: InnerCommandEncoder::list
    /// [`transition_buffers`]: hal::CommandEncoder::transition_buffers
    /// [`transition_textures`]: hal::CommandEncoder::transition_textures
    fn close_and_swap(&mut self) -> Result<(), DeviceError> {
        assert!(self.is_open);
        self.is_open = false;

        let new =
            unsafe { self.raw.end_encoding() }.map_err(|e| self.device.handle_hal_error(e))?;
        self.list.insert(self.list.len() - 1, new);

        Ok(())
    }

    /// Finish the current command buffer and insert it at the beginning
    /// of [`self.list`][l].
    ///
    /// On return, the underlying hal encoder is closed.
    ///
    /// # Panics
    ///
    /// - If the encoder is not open.
    ///
    /// [l]: InnerCommandEncoder::list
    pub(crate) fn close_and_push_front(&mut self) -> Result<(), DeviceError> {
        assert!(self.is_open);
        self.is_open = false;

        let new =
            unsafe { self.raw.end_encoding() }.map_err(|e| self.device.handle_hal_error(e))?;
        self.list.insert(0, new);

        Ok(())
    }

    /// Finish the current command buffer, and push it onto
    /// the end of [`self.list`][l].
    ///
    /// On return, the underlying hal encoder is closed.
    ///
    /// # Panics
    ///
    /// - If the encoder is not open.
    ///
    /// [l]: InnerCommandEncoder::list
    pub(crate) fn close(&mut self) -> Result<(), DeviceError> {
        assert!(self.is_open);
        self.is_open = false;

        let cmd_buf =
            unsafe { self.raw.end_encoding() }.map_err(|e| self.device.handle_hal_error(e))?;
        self.list.push(cmd_buf);

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
    /// [l]: InnerCommandEncoder::list
    fn close_if_open(&mut self) -> Result<(), DeviceError> {
        if self.is_open {
            self.is_open = false;
            let cmd_buf =
                unsafe { self.raw.end_encoding() }.map_err(|e| self.device.handle_hal_error(e))?;
            self.list.push(cmd_buf);
        }

        Ok(())
    }

    /// Begin recording a new command buffer, if we haven't already.
    ///
    /// The underlying hal encoder is put in the "recording" state.
    pub(crate) fn open(&mut self) -> Result<&mut dyn hal::DynCommandEncoder, DeviceError> {
        if !self.is_open {
            self.is_open = true;
            let hal_label = hal_label(Some(self.label.as_str()), self.device.instance_flags);
            unsafe { self.raw.begin_encoding(hal_label) }
                .map_err(|e| self.device.handle_hal_error(e))?;
        }

        Ok(self.raw.as_mut())
    }

    /// Begin recording a new command buffer for a render pass, with
    /// its own label.
    ///
    /// The underlying hal encoder is put in the "recording" state.
    ///
    /// # Panics
    ///
    /// - If the encoder is already open.
    pub(crate) fn open_pass(
        &mut self,
        label: Option<&str>,
    ) -> Result<&mut dyn hal::DynCommandEncoder, DeviceError> {
        assert!(!self.is_open);
        self.is_open = true;

        let hal_label = hal_label(label, self.device.instance_flags);
        unsafe { self.raw.begin_encoding(hal_label) }
            .map_err(|e| self.device.handle_hal_error(e))?;

        Ok(self.raw.as_mut())
    }
}

impl Drop for InnerCommandEncoder {
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
    pub(crate) encoder: InnerCommandEncoder,
    pub(crate) trackers: Tracker,
    pub(crate) temp_resources: Vec<TempResource>,
    pub(crate) indirect_draw_validation_resources: crate::indirect_validation::DrawResources,
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_actions: CommandBufferTextureMemoryActions,
}

/// The mutable state of a [`CommandBuffer`].
pub struct CommandBufferMutable {
    /// The [`wgpu_hal::Api::CommandBuffer`]s we've built so far, and the encoder
    /// they belong to.
    ///
    /// [`wgpu_hal::Api::CommandBuffer`]: hal::Api::CommandBuffer
    pub(crate) encoder: InnerCommandEncoder,

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

    as_actions: Vec<AsAction>,
    temp_resources: Vec<TempResource>,

    indirect_draw_validation_resources: crate::indirect_validation::DrawResources,

    debug_scope_depth: u32,

    #[cfg(feature = "trace")]
    pub(crate) commands: Option<Vec<TraceCommand>>,
}

impl CommandBufferMutable {
    pub(crate) fn open_encoder_and_tracker(
        &mut self,
    ) -> Result<(&mut dyn hal::DynCommandEncoder, &mut Tracker), DeviceError> {
        let encoder = self.encoder.open()?;
        let tracker = &mut self.trackers;

        Ok((encoder, tracker))
    }

    pub(crate) fn into_baked_commands(self) -> BakedCommands {
        BakedCommands {
            encoder: self.encoder,
            trackers: self.trackers,
            temp_resources: self.temp_resources,
            indirect_draw_validation_resources: self.indirect_draw_validation_resources,
            buffer_memory_init_actions: self.buffer_memory_init_actions,
            texture_memory_actions: self.texture_memory_actions,
        }
    }
}

/// A buffer of commands to be submitted to the GPU for execution.
///
/// Once a command buffer is submitted to the queue, its contents are taken
/// to construct a [`BakedCommands`], whose contents eventually become the
/// property of the submission queue.
pub struct CommandBuffer {
    pub(crate) device: Arc<Device>,
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

impl CommandEncoder {
    pub(crate) fn new(
        encoder: Box<dyn hal::DynCommandEncoder>,
        device: &Arc<Device>,
        label: &Label,
    ) -> Self {
        CommandEncoder {
            device: device.clone(),
            label: label.to_string(),
            data: Mutex::new(
                rank::COMMAND_BUFFER_DATA,
                CommandEncoderStatus::Recording(CommandBufferMutable {
                    encoder: InnerCommandEncoder {
                        raw: ManuallyDrop::new(encoder),
                        list: Vec::new(),
                        device: device.clone(),
                        is_open: false,
                        label: label.to_string(),
                    },
                    trackers: Tracker::new(),
                    buffer_memory_init_actions: Default::default(),
                    texture_memory_actions: Default::default(),
                    pending_query_resets: QueryResetMap::new(),
                    as_actions: Default::default(),
                    temp_resources: Default::default(),
                    indirect_draw_validation_resources:
                        crate::indirect_validation::DrawResources::new(device.clone()),
                    debug_scope_depth: 0,
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

    pub(crate) fn new_invalid(
        device: &Arc<Device>,
        label: &Label,
        err: CommandEncoderError,
    ) -> Self {
        CommandEncoder {
            device: device.clone(),
            label: label.to_string(),
            data: Mutex::new(rank::COMMAND_BUFFER_DATA, CommandEncoderStatus::Error(err)),
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
    pub fn take_finished(&self) -> Result<CommandBufferMutable, CommandEncoderError> {
        use CommandEncoderStatus as St;
        match mem::replace(
            &mut *self.data.lock(),
            CommandEncoderStatus::Error(EncoderStateError::Submitted.into()),
        ) {
            St::Finished(command_buffer_mutable) => Ok(command_buffer_mutable),
            St::Error(err) => Err(err),
            St::Recording(_) | St::Locked(_) | St::Consumed | St::Transitioning => unreachable!(),
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
pub struct BasePass<C, E> {
    pub label: Option<String>,

    /// If the pass is invalid, contains the error that caused the invalidation.
    ///
    /// If the pass is valid, this is `None`.
    ///
    /// Passes are serialized into traces. but we don't support doing so for
    /// passes containing errors. These serde attributes allow `E` to be
    /// `Infallible`.
    #[cfg_attr(feature = "serde", serde(skip, default = "Option::default"))]
    pub error: Option<E>,

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

impl<C: Clone, E: Clone> BasePass<C, E> {
    fn new(label: &Label) -> Self {
        Self {
            label: label.as_deref().map(str::to_owned),
            error: None,
            commands: Vec::new(),
            dynamic_offsets: Vec::new(),
            string_data: Vec::new(),
            push_constant_data: Vec::new(),
        }
    }

    fn new_invalid(label: &Label, err: E) -> Self {
        Self {
            label: label.as_deref().map(str::to_owned),
            error: Some(err),
            commands: Vec::new(),
            dynamic_offsets: Vec::new(),
            string_data: Vec::new(),
            push_constant_data: Vec::new(),
        }
    }
}

/// Checks the state of a [`compute::ComputePass`] or [`render::RenderPass`] and
/// evaluates to a mutable reference to the [`BasePass`], if the pass is open and
/// valid.
///
/// If the pass is ended or not valid, **returns from the invoking function**,
/// like the `?` operator.
///
/// If the pass is ended (i.e. the application is attempting to record a command
/// on a finished pass), returns `Err(EncoderStateError::Ended)` from the
/// invoking function, for immediate propagation as a validation error.
///
/// If the pass is open but invalid (i.e. a previous command encountered an
/// error), returns `Ok(())` from the invoking function. The pass should already
/// have stored the previous error, which will be transferred to the parent
/// encoder when the pass is ended, and then raised as a validation error when
/// `finish()` is called for the parent).
///
/// Although in many cases the functionality of `pass_base!` could be achieved
/// by combining a helper method on the passes with the `pass_try!` macro,
/// taking the mutable reference to the base pass in a macro avoids borrowing
/// conflicts when a reference to some other member of the pass struct is
/// needed simultaneously with the base pass reference.
macro_rules! pass_base {
    ($pass:expr, $scope:expr $(,)?) => {
        match (&$pass.parent, &$pass.base.error) {
            // Pass is ended
            (&None, _) => return Err(EncoderStateError::Ended).map_pass_err($scope),
            // Pass is invalid
            (&Some(_), &Some(_)) => return Ok(()),
            // Pass is open and valid
            (&Some(_), &None) => &mut $pass.base,
        }
    };
}
pub(crate) use pass_base;

/// Handles the error case in an expression of type `Result<T, E>`.
///
/// This macro operates like the `?` operator (or, in early Rust versions, the
/// `try!` macro, hence the name `pass_try`). **When there is an error, the
/// macro returns from the invoking function.** However, `Ok(())`, and not the
/// error itself, is returned. The error is stored in the pass and will later be
/// transferred to the parent encoder when the pass ends, and then raised as a
/// validation error when `finish()` is called for the parent.
///
/// `pass_try!` also calls [`MapPassErr::map_pass_err`] to annotate the error
/// with the command being encoded at the time it occurred.
macro_rules! pass_try {
    ($base:expr, $scope:expr, $res:expr $(,)?) => {
        match $res.map_pass_err($scope) {
            Ok(val) => val,
            Err(err) => {
                $base.error.get_or_insert(err);
                return Ok(());
            }
        }
    };
}
pub(crate) use pass_try;

/// Errors related to the state of a command or pass encoder.
///
/// The exact behavior of these errors may change based on the resolution of
/// <https://github.com/gpuweb/gpuweb/issues/5207>.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum EncoderStateError {
    /// Used internally by wgpu functions to indicate the encoder already
    /// contained an error. This variant should usually not be seen by users of
    /// the API, since an effort should be made to provide the caller with a
    /// more specific reason for the encoder being invalid.
    #[error("Encoder is invalid")]
    Invalid,

    /// Returned immediately when an attempt is made to encode a command using
    /// an encoder that has already finished.
    #[error("Encoding must not have ended")]
    Ended,

    /// Returned by a subsequent call to `encoder.finish()`, if there was an
    /// attempt to open a second pass on the encoder while it was locked for
    /// a first pass (i.e. the first pass was still open).
    ///
    /// Note: only command encoders can be locked (not pass encoders).
    #[error("Encoder is locked by a previously created render/compute pass. Before recording any new commands, the pass must be ended.")]
    Locked,

    /// Returned when attempting to end a pass if the parent encoder is not
    /// locked. This can only happen if pass begin/end calls are mismatched.
    #[error(
        "Encoder is not currently locked. A pass can only be ended while the encoder is locked."
    )]
    Unlocked,

    /// The command buffer has already been submitted.
    ///
    /// Although command encoders and command buffers are distinct WebGPU
    /// objects, we use `CommandEncoderStatus` for both.
    #[error("This command buffer has already been submitted.")]
    Submitted,
}

impl WebGpuError for EncoderStateError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            EncoderStateError::Invalid
            | EncoderStateError::Ended
            | EncoderStateError::Locked
            | EncoderStateError::Unlocked
            | EncoderStateError::Submitted => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum CommandEncoderError {
    #[error(transparent)]
    State(#[from] EncoderStateError),
    #[error(transparent)]
    Device(#[from] DeviceError),
    #[error(transparent)]
    InvalidResource(#[from] InvalidResourceError),
    #[error(transparent)]
    DestroyedResource(#[from] DestroyedResourceError),
    #[error(transparent)]
    ResourceUsage(#[from] ResourceUsageCompatibilityError),
    #[error(transparent)]
    DebugGroupError(#[from] DebugGroupError),
    #[error(transparent)]
    MissingFeatures(#[from] MissingFeatures),
    #[error(transparent)]
    Transfer(#[from] TransferError),
    #[error(transparent)]
    Clear(#[from] ClearError),
    #[error(transparent)]
    Query(#[from] QueryError),
    #[error(transparent)]
    BuildAccelerationStructure(#[from] BuildAccelerationStructureError),
    #[error(transparent)]
    TransitionResources(#[from] TransitionResourcesError),
    #[error(transparent)]
    ComputePass(#[from] ComputePassError),
    #[error(transparent)]
    RenderPass(#[from] RenderPassError),
}

impl CommandEncoderError {
    fn is_destroyed_error(&self) -> bool {
        matches!(
            self,
            Self::DestroyedResource(_)
                | Self::Clear(ClearError::DestroyedResource(_))
                | Self::Query(QueryError::DestroyedResource(_))
                | Self::ComputePass(ComputePassError {
                    inner: ComputePassErrorInner::DestroyedResource(_),
                    ..
                })
                | Self::RenderPass(RenderPassError {
                    inner: RenderPassErrorInner::DestroyedResource(_),
                    ..
                })
                | Self::RenderPass(RenderPassError {
                    inner: RenderPassErrorInner::RenderCommand(
                        RenderCommandError::DestroyedResource(_)
                    ),
                    ..
                })
                | Self::RenderPass(RenderPassError {
                    inner: RenderPassErrorInner::RenderCommand(RenderCommandError::BindingError(
                        BindingError::DestroyedResource(_)
                    )),
                    ..
                })
        )
    }
}

impl WebGpuError for CommandEncoderError {
    fn webgpu_error_type(&self) -> ErrorType {
        let e: &dyn WebGpuError = match self {
            Self::Device(e) => e,
            Self::InvalidResource(e) => e,
            Self::DebugGroupError(e) => e,
            Self::MissingFeatures(e) => e,
            Self::State(e) => e,
            Self::DestroyedResource(e) => e,
            Self::Transfer(e) => e,
            Self::Clear(e) => e,
            Self::Query(e) => e,
            Self::BuildAccelerationStructure(e) => e,
            Self::TransitionResources(e) => e,
            Self::ResourceUsage(e) => e,
            Self::ComputePass(e) => e,
            Self::RenderPass(e) => e,
        };
        e.webgpu_error_type()
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum DebugGroupError {
    #[error("Cannot pop debug group, because number of pushed debug groups is zero")]
    InvalidPop,
    #[error("A debug group was not popped before the encoder was finished")]
    MissingPop,
}

impl WebGpuError for DebugGroupError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::InvalidPop | Self::MissingPop => ErrorType::Validation,
        }
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum TimestampWritesError {
    #[error(
        "begin and end indices of pass timestamp writes are both set to {idx}, which is not allowed"
    )]
    IndicesEqual { idx: u32 },
    #[error("no begin or end indices were specified for pass timestamp writes, expected at least one to be set")]
    IndicesMissing,
}

impl WebGpuError for TimestampWritesError {
    fn webgpu_error_type(&self) -> ErrorType {
        match self {
            Self::IndicesEqual { .. } | Self::IndicesMissing => ErrorType::Validation,
        }
    }
}

impl Global {
    pub fn command_encoder_finish(
        &self,
        encoder_id: id::CommandEncoderId,
        desc: &wgt::CommandBufferDescriptor<Label>,
        id_in: Option<id::CommandBufferId>,
    ) -> (id::CommandBufferId, Option<CommandEncoderError>) {
        profiling::scope!("CommandEncoder::finish");

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);

        let data = cmd_enc.data.lock().finish();

        // Errors related to destroyed resources are not reported until the
        // command buffer is submitted.
        let error = match data {
            CommandEncoderStatus::Error(ref e) if !e.is_destroyed_error() => Some(e.clone()),
            _ => None,
        };

        let cmd_buf = CommandBuffer {
            device: cmd_enc.device.clone(),
            label: desc.label.to_string(),
            data: Mutex::new(rank::COMMAND_BUFFER_DATA, data),
        };

        let cmd_buf_id = hub.command_buffers.prepare(id_in).assign(Arc::new(cmd_buf));

        (cmd_buf_id, error)
    }

    pub fn command_encoder_push_debug_group(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::push_debug_group");
        api_log!("CommandEncoder::push_debug_group {label}");

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            cmd_buf_data.debug_scope_depth += 1;

            #[cfg(feature = "trace")]
            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(TraceCommand::PushDebugGroup(label.to_owned()));
            }

            cmd_enc.device.check_is_valid()?;

            let cmd_buf_raw = cmd_buf_data.encoder.open()?;
            if !cmd_enc
                .device
                .instance_flags
                .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
            {
                unsafe {
                    cmd_buf_raw.begin_debug_marker(label);
                }
            }

            Ok(())
        })
    }

    pub fn command_encoder_insert_debug_marker(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::insert_debug_marker");
        api_log!("CommandEncoder::insert_debug_marker {label}");

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            #[cfg(feature = "trace")]
            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(TraceCommand::InsertDebugMarker(label.to_owned()));
            }

            cmd_enc.device.check_is_valid()?;

            if !cmd_enc
                .device
                .instance_flags
                .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
            {
                let cmd_buf_raw = cmd_buf_data.encoder.open()?;
                unsafe {
                    cmd_buf_raw.insert_debug_marker(label);
                }
            }

            Ok(())
        })
    }

    pub fn command_encoder_pop_debug_group(
        &self,
        encoder_id: id::CommandEncoderId,
    ) -> Result<(), EncoderStateError> {
        profiling::scope!("CommandEncoder::pop_debug_marker");
        api_log!("CommandEncoder::pop_debug_group");

        let hub = &self.hub;

        let cmd_enc = hub.command_encoders.get(encoder_id);
        let mut cmd_buf_data = cmd_enc.data.lock();
        cmd_buf_data.record_with(|cmd_buf_data| -> Result<(), CommandEncoderError> {
            if cmd_buf_data.debug_scope_depth == 0 {
                return Err(DebugGroupError::InvalidPop.into());
            }
            cmd_buf_data.debug_scope_depth -= 1;

            #[cfg(feature = "trace")]
            if let Some(ref mut list) = cmd_buf_data.commands {
                list.push(TraceCommand::PopDebugGroup);
            }

            cmd_enc.device.check_is_valid()?;

            let cmd_buf_raw = cmd_buf_data.encoder.open()?;
            if !cmd_enc
                .device
                .instance_flags
                .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
            {
                unsafe {
                    cmd_buf_raw.end_debug_marker();
                }
            }

            Ok(())
        })
    }

    fn validate_pass_timestamp_writes<E>(
        device: &Device,
        query_sets: &Storage<Fallible<QuerySet>>,
        timestamp_writes: &PassTimestampWrites,
    ) -> Result<ArcPassTimestampWrites, E>
    where
        E: From<TimestampWritesError>
            + From<QueryUseError>
            + From<DeviceError>
            + From<MissingFeatures>
            + From<InvalidResourceError>,
    {
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
                return Err(TimestampWritesError::IndicesEqual { idx: begin }.into());
            }
        }

        if beginning_of_pass_write_index
            .or(end_of_pass_write_index)
            .is_none()
        {
            return Err(TimestampWritesError::IndicesMissing.into());
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

/// Helper to attach [`PassErrorScope`] to errors.
trait MapPassErr<T> {
    fn map_pass_err(self, scope: PassErrorScope) -> T;
}

impl<T, E, F> MapPassErr<Result<T, F>> for Result<T, E>
where
    E: MapPassErr<F>,
{
    fn map_pass_err(self, scope: PassErrorScope) -> Result<T, F> {
        self.map_err(|err| err.map_pass_err(scope))
    }
}

impl MapPassErr<PassStateError> for EncoderStateError {
    fn map_pass_err(self, scope: PassErrorScope) -> PassStateError {
        PassStateError { scope, inner: self }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum DrawKind {
    Draw,
    DrawIndirect,
    MultiDrawIndirect,
    MultiDrawIndirectCount,
}

/// The type of draw command(indexed or not, or mesh shader)
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DrawCommandFamily {
    Draw,
    DrawIndexed,
    DrawMeshTasks,
}

/// A command that can be recorded in a pass or bundle.
///
/// This is used to provide context for errors during command recording.
/// [`MapPassErr`] is used as a helper to attach a `PassErrorScope` to
/// an error.
///
/// The [`PassErrorScope::Bundle`] and [`PassErrorScope::Pass`] variants
/// are used when the error occurs during the opening or closing of the
/// pass or bundle.
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
    Draw {
        kind: DrawKind,
        family: DrawCommandFamily,
    },
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

/// Variant of `EncoderStateError` that includes the pass scope.
#[derive(Clone, Debug, Error)]
#[error("{scope}")]
pub struct PassStateError {
    pub scope: PassErrorScope,
    #[source]
    pub(super) inner: EncoderStateError,
}

impl WebGpuError for PassStateError {
    fn webgpu_error_type(&self) -> ErrorType {
        let Self { scope: _, inner } = self;
        inner.webgpu_error_type()
    }
}

mod bind;
mod bundle;
mod clear;
mod compute;
mod draw;
mod memory_init;
mod query;
mod render;
mod transfer;

use std::slice;

pub(crate) use self::clear::clear_texture;
pub use self::{
    bundle::*, clear::ClearError, compute::*, draw::*, query::*, render::*, transfer::*,
};

use self::memory_init::CommandBufferTextureMemoryActions;

use crate::error::{ErrorFormatter, PrettyError};
use crate::init_tracker::BufferInitTrackerAction;
use crate::track::{Tracker, UsageScope};
use crate::{
    global::Global,
    hal_api::HalApi,
    hub::Token,
    id,
    identity::GlobalIdentityHandlerFactory,
    resource::{Buffer, Texture},
    storage::Storage,
    Label, Stored,
};

use hal::CommandEncoder as _;
use thiserror::Error;

#[cfg(feature = "trace")]
use crate::device::trace::Command as TraceCommand;

const PUSH_CONSTANT_CLEAR_ARRAY: &[u32] = &[0_u32; 64];

#[derive(Debug)]
enum CommandEncoderStatus {
    Recording,
    Finished,
    Error,
}

struct CommandEncoder<A: hal::Api> {
    raw: A::CommandEncoder,
    list: Vec<A::CommandBuffer>,
    is_open: bool,
    label: Option<String>,
}

//TODO: handle errors better
impl<A: hal::Api> CommandEncoder<A> {
    /// Closes the live encoder
    fn close_and_swap(&mut self) {
        if self.is_open {
            self.is_open = false;
            let new = unsafe { self.raw.end_encoding().unwrap() };
            self.list.insert(self.list.len() - 1, new);
        }
    }

    fn close(&mut self) {
        if self.is_open {
            self.is_open = false;
            let cmd_buf = unsafe { self.raw.end_encoding().unwrap() };
            self.list.push(cmd_buf);
        }
    }

    fn discard(&mut self) {
        if self.is_open {
            self.is_open = false;
            unsafe { self.raw.discard_encoding() };
        }
    }

    fn open(&mut self) -> &mut A::CommandEncoder {
        if !self.is_open {
            self.is_open = true;
            let label = self.label.as_deref();
            unsafe { self.raw.begin_encoding(label).unwrap() };
        }
        &mut self.raw
    }

    fn open_pass(&mut self, label: Option<&str>) {
        self.is_open = true;
        unsafe { self.raw.begin_encoding(label).unwrap() };
    }
}

pub struct BakedCommands<A: HalApi> {
    pub(crate) encoder: A::CommandEncoder,
    pub(crate) list: Vec<A::CommandBuffer>,
    pub(crate) trackers: Tracker<A>,
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_actions: CommandBufferTextureMemoryActions,
}

pub(crate) struct DestroyedBufferError(pub id::BufferId);
pub(crate) struct DestroyedTextureError(pub id::TextureId);

pub struct CommandBuffer<A: HalApi> {
    encoder: CommandEncoder<A>,
    status: CommandEncoderStatus,
    pub(crate) device_id: Stored<id::DeviceId>,
    pub(crate) trackers: Tracker<A>,
    buffer_memory_init_actions: Vec<BufferInitTrackerAction>,
    texture_memory_actions: CommandBufferTextureMemoryActions,
    pub(crate) pending_query_resets: QueryResetMap<A>,
    limits: wgt::Limits,
    support_clear_texture: bool,
    #[cfg(feature = "trace")]
    pub(crate) commands: Option<Vec<TraceCommand>>,
}

impl<A: HalApi> CommandBuffer<A> {
    pub(crate) fn new(
        encoder: A::CommandEncoder,
        device_id: Stored<id::DeviceId>,
        limits: wgt::Limits,
        _downlevel: wgt::DownlevelCapabilities,
        features: wgt::Features,
        #[cfg(feature = "trace")] enable_tracing: bool,
        label: Option<String>,
    ) -> Self {
        CommandBuffer {
            encoder: CommandEncoder {
                raw: encoder,
                is_open: false,
                list: Vec::new(),
                label,
            },
            status: CommandEncoderStatus::Recording,
            device_id,
            trackers: Tracker::new(),
            buffer_memory_init_actions: Default::default(),
            texture_memory_actions: Default::default(),
            pending_query_resets: QueryResetMap::new(),
            limits,
            support_clear_texture: features.contains(wgt::Features::CLEAR_TEXTURE),
            #[cfg(feature = "trace")]
            commands: if enable_tracing {
                Some(Vec::new())
            } else {
                None
            },
        }
    }

    pub(crate) fn insert_barriers_from_tracker(
        raw: &mut A::CommandEncoder,
        base: &mut Tracker<A>,
        head: &Tracker<A>,
        buffer_guard: &Storage<Buffer<A>, id::BufferId>,
        texture_guard: &Storage<Texture<A>, id::TextureId>,
    ) {
        profiling::scope!("insert_barriers");

        base.buffers.set_from_tracker(&head.buffers);
        base.textures
            .set_from_tracker(texture_guard, &head.textures);

        Self::drain_barriers(raw, base, buffer_guard, texture_guard);
    }

    pub(crate) fn insert_barriers_from_scope(
        raw: &mut A::CommandEncoder,
        base: &mut Tracker<A>,
        head: &UsageScope<A>,
        buffer_guard: &Storage<Buffer<A>, id::BufferId>,
        texture_guard: &Storage<Texture<A>, id::TextureId>,
    ) {
        profiling::scope!("insert_barriers");

        base.buffers.set_from_usage_scope(&head.buffers);
        base.textures
            .set_from_usage_scope(texture_guard, &head.textures);

        Self::drain_barriers(raw, base, buffer_guard, texture_guard);
    }

    pub(crate) fn drain_barriers(
        raw: &mut A::CommandEncoder,
        base: &mut Tracker<A>,
        buffer_guard: &Storage<Buffer<A>, id::BufferId>,
        texture_guard: &Storage<Texture<A>, id::TextureId>,
    ) {
        profiling::scope!("drain_barriers");

        let buffer_barriers = base.buffers.drain().map(|pending| {
            let buf = unsafe { &buffer_guard.get_unchecked(pending.id) };
            pending.into_hal(buf)
        });
        let texture_barriers = base.textures.drain().map(|pending| {
            let tex = unsafe { texture_guard.get_unchecked(pending.id) };
            pending.into_hal(tex)
        });

        unsafe {
            raw.transition_buffers(buffer_barriers);
            raw.transition_textures(texture_barriers);
        }
    }
}

impl<A: HalApi> CommandBuffer<A> {
    fn get_encoder_mut(
        storage: &mut Storage<Self, id::CommandEncoderId>,
        id: id::CommandEncoderId,
    ) -> Result<&mut Self, CommandEncoderError> {
        match storage.get_mut(id) {
            Ok(cmd_buf) => match cmd_buf.status {
                CommandEncoderStatus::Recording => Ok(cmd_buf),
                CommandEncoderStatus::Finished => Err(CommandEncoderError::NotRecording),
                CommandEncoderStatus::Error => Err(CommandEncoderError::Invalid),
            },
            Err(_) => Err(CommandEncoderError::Invalid),
        }
    }

    pub fn is_finished(&self) -> bool {
        match self.status {
            CommandEncoderStatus::Finished => true,
            _ => false,
        }
    }

    pub(crate) fn into_baked(self) -> BakedCommands<A> {
        BakedCommands {
            encoder: self.encoder.raw,
            list: self.encoder.list,
            trackers: self.trackers,
            buffer_memory_init_actions: self.buffer_memory_init_actions,
            texture_memory_actions: self.texture_memory_actions,
        }
    }
}

impl<A: HalApi> crate::resource::Resource for CommandBuffer<A> {
    const TYPE: &'static str = "CommandBuffer";

    fn life_guard(&self) -> &crate::LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        self.encoder.label.as_ref().map_or("", |s| s.as_str())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BasePassRef<'a, C> {
    pub label: Option<&'a str>,
    pub commands: &'a [C],
    pub dynamic_offsets: &'a [wgt::DynamicOffset],
    pub string_data: &'a [u8],
    pub push_constant_data: &'a [u32],
}

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
#[derive(Debug)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "trace"),
    derive(serde::Serialize)
)]
#[cfg_attr(
    any(feature = "serial-pass", feature = "replay"),
    derive(serde::Deserialize)
)]
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

    #[cfg(feature = "trace")]
    fn from_ref(base: BasePassRef<C>) -> Self {
        Self {
            label: base.label.map(str::to_string),
            commands: base.commands.to_vec(),
            dynamic_offsets: base.dynamic_offsets.to_vec(),
            string_data: base.string_data.to_vec(),
            push_constant_data: base.push_constant_data.to_vec(),
        }
    }

    pub fn as_ref(&self) -> BasePassRef<C> {
        BasePassRef {
            label: self.label.as_deref(),
            commands: &self.commands,
            dynamic_offsets: &self.dynamic_offsets,
            string_data: &self.string_data,
            push_constant_data: &self.push_constant_data,
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
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_finish<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        _desc: &wgt::CommandBufferDescriptor<Label>,
    ) -> (id::CommandBufferId, Option<CommandEncoderError>) {
        profiling::scope!("CommandEncoder::finish");

        let hub = A::hub(self);
        let mut token = Token::root();
        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);

        let error = match cmd_buf_guard.get_mut(encoder_id) {
            Ok(cmd_buf) => match cmd_buf.status {
                CommandEncoderStatus::Recording => {
                    cmd_buf.encoder.close();
                    cmd_buf.status = CommandEncoderStatus::Finished;
                    //Note: if we want to stop tracking the swapchain texture view,
                    // this is the place to do it.
                    log::trace!("Command buffer {:?}", encoder_id);
                    None
                }
                CommandEncoderStatus::Finished => Some(CommandEncoderError::NotRecording),
                CommandEncoderStatus::Error => {
                    cmd_buf.encoder.discard();
                    Some(CommandEncoderError::Invalid)
                }
            },
            Err(_) => Some(CommandEncoderError::Invalid),
        };

        (encoder_id, error)
    }

    pub fn command_encoder_push_debug_group<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::push_debug_group");
        log::trace!("CommandEncoder::push_debug_group {label}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::PushDebugGroup(label.to_string()));
        }

        let cmd_buf_raw = cmd_buf.encoder.open();
        if !self
            .instance
            .flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
        {
            unsafe {
                cmd_buf_raw.begin_debug_marker(label);
            }
        }
        Ok(())
    }

    pub fn command_encoder_insert_debug_marker<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::insert_debug_marker");
        log::trace!("CommandEncoder::insert_debug_marker {label}");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::InsertDebugMarker(label.to_string()));
        }

        if !self
            .instance
            .flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
        {
            let cmd_buf_raw = cmd_buf.encoder.open();
            unsafe {
                cmd_buf_raw.insert_debug_marker(label);
            }
        }
        Ok(())
    }

    pub fn command_encoder_pop_debug_group<A: HalApi>(
        &self,
        encoder_id: id::CommandEncoderId,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::pop_debug_marker");
        log::trace!("CommandEncoder::pop_debug_group");

        let hub = A::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id)?;

        #[cfg(feature = "trace")]
        if let Some(ref mut list) = cmd_buf.commands {
            list.push(TraceCommand::PopDebugGroup);
        }

        let cmd_buf_raw = cmd_buf.encoder.open();
        if !self
            .instance
            .flags
            .contains(wgt::InstanceFlags::DISCARD_HAL_LABELS)
        {
            unsafe {
                cmd_buf_raw.end_debug_marker();
            }
        }
        Ok(())
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
    last_states: [StateChange<id::BindGroupId>; hal::MAX_BIND_GROUPS],
}

impl BindGroupStateChange {
    fn new() -> Self {
        Self {
            last_states: [StateChange::new(); hal::MAX_BIND_GROUPS],
        }
    }

    unsafe fn set_and_check_redundant(
        &mut self,
        bind_group_id: id::BindGroupId,
        index: u32,
        dynamic_offsets: &mut Vec<u32>,
        offsets: *const wgt::DynamicOffset,
        offset_length: usize,
    ) -> bool {
        // For now never deduplicate bind groups with dynamic offsets.
        if offset_length == 0 {
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
            dynamic_offsets
                .extend_from_slice(unsafe { slice::from_raw_parts(offsets, offset_length) });
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

#[derive(Clone, Copy, Debug, Error)]
pub enum PassErrorScope {
    #[error("In a bundle parameter")]
    Bundle,
    #[error("In a pass parameter")]
    Pass(id::CommandEncoderId),
    #[error("In a set_bind_group command")]
    SetBindGroup(id::BindGroupId),
    #[error("In a set_pipeline command")]
    SetPipelineRender(id::RenderPipelineId),
    #[error("In a set_pipeline command")]
    SetPipelineCompute(id::ComputePipelineId),
    #[error("In a set_push_constant command")]
    SetPushConstant,
    #[error("In a set_vertex_buffer command")]
    SetVertexBuffer(id::BufferId),
    #[error("In a set_index_buffer command")]
    SetIndexBuffer(id::BufferId),
    #[error("In a set_viewport command")]
    SetViewport,
    #[error("In a set_scissor_rect command")]
    SetScissorRect,
    #[error("In a draw command, indexed:{indexed} indirect:{indirect}")]
    Draw {
        indexed: bool,
        indirect: bool,
        pipeline: Option<id::RenderPipelineId>,
    },
    #[error("While resetting queries after the renderpass was ran")]
    QueryReset,
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
    Dispatch {
        indirect: bool,
        pipeline: Option<id::ComputePipelineId>,
    },
    #[error("In a pop_debug_group command")]
    PopDebugGroup,
}

impl PrettyError for PassErrorScope {
    fn fmt_pretty(&self, fmt: &mut ErrorFormatter) {
        // This error is not in the error chain, only notes are needed
        match *self {
            Self::Pass(id) => {
                fmt.command_buffer_label(&id);
            }
            Self::SetBindGroup(id) => {
                fmt.bind_group_label(&id);
            }
            Self::SetPipelineRender(id) => {
                fmt.render_pipeline_label(&id);
            }
            Self::SetPipelineCompute(id) => {
                fmt.compute_pipeline_label(&id);
            }
            Self::SetVertexBuffer(id) => {
                fmt.buffer_label(&id);
            }
            Self::SetIndexBuffer(id) => {
                fmt.buffer_label(&id);
            }
            Self::Draw {
                pipeline: Some(id), ..
            } => {
                fmt.render_pipeline_label(&id);
            }
            Self::Dispatch {
                pipeline: Some(id), ..
            } => {
                fmt.compute_pipeline_label(&id);
            }
            _ => {}
        }
    }
}

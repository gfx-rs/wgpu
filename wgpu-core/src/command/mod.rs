/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod allocator;
mod bind;
mod bundle;
mod compute;
mod draw;
mod query;
mod render;
mod transfer;

pub(crate) use self::allocator::CommandAllocator;
pub use self::allocator::CommandAllocatorError;
pub use self::bundle::*;
pub use self::compute::*;
pub use self::draw::*;
pub use self::query::*;
pub use self::render::*;
pub use self::transfer::*;

use crate::{
    device::{all_buffer_stages, all_image_stages},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id,
    memory_init_tracker::MemoryInitTrackerAction,
    resource::{Buffer, Texture},
    track::TrackerSet,
    Label, PrivateFeatures, Stored,
};

use hal::command::CommandBuffer as _;
use smallvec::SmallVec;
use thiserror::Error;

use std::thread::ThreadId;

const PUSH_CONSTANT_CLEAR_ARRAY: &[u32] = &[0_u32; 64];

#[derive(Debug)]
pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Vec<B::CommandBuffer>,
    is_recording: bool,
    recorded_thread_id: ThreadId,
    pub(crate) device_id: Stored<id::DeviceId>,
    pub(crate) trackers: TrackerSet,
    pub(crate) used_swap_chains: SmallVec<[Stored<id::SwapChainId>; 1]>,
    pub(crate) buffer_memory_init_actions: Vec<MemoryInitTrackerAction<id::BufferId>>,
    limits: wgt::Limits,
    downlevel: wgt::DownlevelProperties,
    private_features: PrivateFeatures,
    has_labels: bool,
    #[cfg(feature = "trace")]
    pub(crate) commands: Option<Vec<crate::device::trace::Command>>,
    #[cfg(debug_assertions)]
    pub(crate) label: String,
}

impl<B: GfxBackend> CommandBuffer<B> {
    fn get_encoder_mut(
        storage: &mut Storage<Self, id::CommandEncoderId>,
        id: id::CommandEncoderId,
    ) -> Result<&mut Self, CommandEncoderError> {
        match storage.get_mut(id) {
            Ok(cmd_buf) if cmd_buf.is_recording => Ok(cmd_buf),
            Ok(_) => Err(CommandEncoderError::NotRecording),
            Err(_) => Err(CommandEncoderError::Invalid),
        }
    }

    pub(crate) fn insert_barriers(
        raw: &mut B::CommandBuffer,
        base: &mut TrackerSet,
        head: &TrackerSet,
        buffer_guard: &Storage<Buffer<B>, id::BufferId>,
        texture_guard: &Storage<Texture<B>, id::TextureId>,
    ) {
        use hal::command::CommandBuffer as _;

        debug_assert_eq!(B::VARIANT, base.backend());
        debug_assert_eq!(B::VARIANT, head.backend());

        let buffer_barriers = base.buffers.merge_replace(&head.buffers).map(|pending| {
            let buf = &buffer_guard[pending.id];
            pending.into_hal(buf)
        });
        let texture_barriers = base.textures.merge_replace(&head.textures).map(|pending| {
            let tex = &texture_guard[pending.id];
            pending.into_hal(tex)
        });
        base.views.merge_extend(&head.views).unwrap();
        base.bind_groups.merge_extend(&head.bind_groups).unwrap();
        base.samplers.merge_extend(&head.samplers).unwrap();
        base.compute_pipes
            .merge_extend(&head.compute_pipes)
            .unwrap();
        base.render_pipes.merge_extend(&head.render_pipes).unwrap();
        base.bundles.merge_extend(&head.bundles).unwrap();

        let stages = all_buffer_stages() | all_image_stages();
        unsafe {
            raw.pipeline_barrier(
                stages..stages,
                hal::memory::Dependencies::empty(),
                buffer_barriers.chain(texture_barriers),
            );
        }
    }
}

impl<B: hal::Backend> crate::hub::Resource for CommandBuffer<B> {
    const TYPE: &'static str = "CommandBuffer";

    fn life_guard(&self) -> &crate::LifeGuard {
        unreachable!()
    }

    fn label(&self) -> &str {
        #[cfg(debug_assertions)]
        return &self.label;
        #[cfg(not(debug_assertions))]
        return "";
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
    pub commands: Vec<C>,
    pub dynamic_offsets: Vec<wgt::DynamicOffset>,
    pub string_data: Vec<u8>,
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
pub enum CommandEncoderError {
    #[error("command encoder is invalid")]
    Invalid,
    #[error("command encoder must be active")]
    NotRecording,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_finish<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        _desc: &wgt::CommandBufferDescriptor<Label>,
    ) -> (id::CommandBufferId, Option<CommandEncoderError>) {
        profiling::scope!("CommandEncoder::finish");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (swap_chain_guard, mut token) = hub.swap_chains.read(&mut token);
        //TODO: actually close the last recorded command buffer
        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);

        let error = match CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id) {
            Ok(cmd_buf) => {
                cmd_buf.is_recording = false;
                // stop tracking the swapchain image, if used
                for sc_id in cmd_buf.used_swap_chains.iter() {
                    let view_id = swap_chain_guard[sc_id.value]
                        .acquired_view_id
                        .as_ref()
                        .expect("Used swap chain frame has already presented");
                    cmd_buf.trackers.views.remove(view_id.value);
                }
                log::trace!("Command buffer {:?} {:#?}", encoder_id, cmd_buf.trackers);
                None
            }
            Err(e) => Some(e),
        };

        (encoder_id, error)
    }

    pub fn command_encoder_push_debug_group<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::push_debug_group");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id)?;
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();

        unsafe {
            cmd_buf_raw.begin_debug_marker(label, 0);
        }
        Ok(())
    }

    pub fn command_encoder_insert_debug_marker<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::insert_debug_marker");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id)?;
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();

        unsafe {
            cmd_buf_raw.insert_debug_marker(label, 0);
        }
        Ok(())
    }

    pub fn command_encoder_pop_debug_group<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
    ) -> Result<(), CommandEncoderError> {
        profiling::scope!("CommandEncoder::pop_debug_marker");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmd_buf_guard, _) = hub.command_buffers.write(&mut token);
        let cmd_buf = CommandBuffer::get_encoder_mut(&mut *cmd_buf_guard, encoder_id)?;
        let cmd_buf_raw = cmd_buf.raw.last_mut().unwrap();

        unsafe {
            cmd_buf_raw.end_debug_marker();
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

#[derive(Debug)]
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
    fn is_unset(&self) -> bool {
        self.last_state.is_none()
    }
    fn reset(&mut self) {
        self.last_state = None;
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

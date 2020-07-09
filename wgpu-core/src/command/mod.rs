/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod allocator;
mod bind;
mod bundle;
mod compute;
mod render;
mod transfer;

pub(crate) use self::allocator::CommandAllocator;
pub use self::bundle::*;
pub use self::compute::*;
pub use self::render::*;
pub use self::transfer::*;

use crate::{
    device::{all_buffer_stages, all_image_stages},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id,
    resource::{Buffer, Texture},
    span,
    track::TrackerSet,
    PrivateFeatures, Stored,
};

use hal::command::CommandBuffer as _;

use std::thread::ThreadId;

const PUSH_CONSTANT_CLEAR_ARRAY: &[u32] = &[0_u32; 64];

#[derive(Debug)]
pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Vec<B::CommandBuffer>,
    is_recording: bool,
    recorded_thread_id: ThreadId,
    pub(crate) device_id: Stored<id::DeviceId>,
    pub(crate) trackers: TrackerSet,
    pub(crate) used_swap_chain: Option<(Stored<id::SwapChainId>, B::Framebuffer)>,
    limits: wgt::Limits,
    private_features: PrivateFeatures,
    #[cfg(feature = "trace")]
    pub(crate) commands: Option<Vec<crate::device::trace::Command>>,
}

impl<B: GfxBackend> CommandBuffer<B> {
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

#[derive(Copy, Clone, Debug)]
pub struct BasePassRef<'a, C> {
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
    pub commands: Vec<C>,
    pub dynamic_offsets: Vec<wgt::DynamicOffset>,
    pub string_data: Vec<u8>,
    pub push_constant_data: Vec<u32>,
}

impl<C: Clone> BasePass<C> {
    fn new() -> Self {
        BasePass {
            commands: Vec::new(),
            dynamic_offsets: Vec::new(),
            string_data: Vec::new(),
            push_constant_data: Vec::new(),
        }
    }

    #[cfg(feature = "trace")]
    fn from_ref(base: BasePassRef<C>) -> Self {
        BasePass {
            commands: base.commands.to_vec(),
            dynamic_offsets: base.dynamic_offsets.to_vec(),
            string_data: base.string_data.to_vec(),
            push_constant_data: base.push_constant_data.to_vec(),
        }
    }

    pub fn as_ref(&self) -> BasePassRef<C> {
        BasePassRef {
            commands: &self.commands,
            dynamic_offsets: &self.dynamic_offsets,
            string_data: &self.string_data,
            push_constant_data: &self.push_constant_data,
        }
    }
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_finish<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        _desc: &wgt::CommandBufferDescriptor,
    ) -> id::CommandBufferId {
        span!(_guard, INFO, "CommandEncoder::finish");

        let hub = B::hub(self);
        let mut token = Token::root();
        let (swap_chain_guard, mut token) = hub.swap_chains.read(&mut token);
        //TODO: actually close the last recorded command buffer
        let (mut comb_guard, _) = hub.command_buffers.write(&mut token);
        let comb = &mut comb_guard[encoder_id];
        assert!(comb.is_recording, "Command buffer must be recording");
        comb.is_recording = false;
        // stop tracking the swapchain image, if used
        if let Some((ref sc_id, _)) = comb.used_swap_chain {
            let view_id = swap_chain_guard[sc_id.value]
                .acquired_view_id
                .as_ref()
                .expect("Used swap chain frame has already presented");
            comb.trackers.views.remove(view_id.value);
        }
        log::trace!("Command buffer {:?} {:#?}", encoder_id, comb.trackers);
        encoder_id
    }

    pub fn command_encoder_push_debug_group<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) {
        span!(_guard, DEBUG, "CommandEncoder::push_debug_group");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, _) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];
        let cmb_raw = cmb.raw.last_mut().unwrap();

        unsafe {
            cmb_raw.begin_debug_marker(label, 0);
        }
    }

    pub fn command_encoder_insert_debug_marker<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) {
        span!(_guard, DEBUG, "CommandEncoder::insert_debug_marker");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, _) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];
        let cmb_raw = cmb.raw.last_mut().unwrap();

        unsafe {
            cmb_raw.insert_debug_marker(label, 0);
        }
    }

    pub fn command_encoder_pop_debug_group<B: GfxBackend>(&self, encoder_id: id::CommandEncoderId) {
        span!(_guard, DEBUG, "CommandEncoder::pop_debug_marker");

        let hub = B::hub(self);
        let mut token = Token::root();

        let (mut cmb_guard, _) = hub.command_buffers.write(&mut token);
        let cmb = &mut cmb_guard[encoder_id];
        let cmb_raw = cmb.raw.last_mut().unwrap();

        unsafe {
            cmb_raw.end_debug_marker();
        }
    }
}

fn push_constant_clear<PushFn>(offset: u32, size_bytes: u32, mut push_fn: PushFn)
where
    PushFn: FnMut(u32, &[u32]),
{
    let mut count_words = 0_u32;
    let size_words = size_bytes / 4;
    while count_words < size_words {
        let count_bytes = count_words * 4;
        let size_to_write_words =
            (size_words - count_words).min(PUSH_CONSTANT_CLEAR_ARRAY.len() as u32);

        push_fn(
            offset + count_bytes,
            &PUSH_CONSTANT_CLEAR_ARRAY[0..size_to_write_words as usize],
        );

        count_words += size_to_write_words;
    }
}

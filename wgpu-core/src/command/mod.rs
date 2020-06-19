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
    device::{all_buffer_stages, all_image_stages, MAX_COLOR_TARGETS},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id,
    resource::{Buffer, Texture},
    track::TrackerSet,
    PrivateFeatures, Stored,
};

use hal::command::CommandBuffer as _;

use peek_poke::PeekPoke;

use std::{marker::PhantomData, mem, ptr, slice, thread::ThreadId};

#[derive(Clone, Copy, Debug, PeekPoke)]
pub struct PhantomSlice<T>(PhantomData<T>);

impl<T> Default for PhantomSlice<T> {
    fn default() -> Self {
        PhantomSlice(PhantomData)
    }
}

impl<T> PhantomSlice<T> {
    unsafe fn decode_unaligned<'a>(
        self,
        pointer: *const u8,
        count: usize,
        bound: *const u8,
    ) -> (*const u8, &'a [T]) {
        let align_offset = pointer.align_offset(mem::align_of::<T>());
        let aligned = pointer.add(align_offset);
        let size = count * mem::size_of::<T>();
        let end = aligned.add(size);
        assert!(
            end <= bound,
            "End of phantom slice ({:?}) exceeds bound ({:?})",
            end,
            bound
        );
        (end, slice::from_raw_parts(aligned as *const T, count))
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct RawPass<P> {
    data: *mut u8,
    base: *mut u8,
    capacity: usize,
    parent: P,
}

impl<P: Copy> RawPass<P> {
    fn new<T>(parent: P) -> Self {
        let mut vec = Vec::<T>::with_capacity(1);
        let ptr = vec.as_mut_ptr() as *mut u8;
        let capacity = mem::size_of::<T>();
        assert_ne!(capacity, 0);
        mem::forget(vec);
        RawPass {
            data: ptr,
            base: ptr,
            capacity,
            parent,
        }
    }

    /// Finish encoding a raw pass.
    ///
    /// The last command is provided, yet the encoder
    /// is guaranteed to have exactly `C::max_size()` space for it.
    unsafe fn finish<C: peek_poke::Poke>(&mut self, command: C) {
        self.ensure_extra_size(C::max_size());
        let extended_end = self.data.add(C::max_size());
        let end = command.poke_into(self.data);
        ptr::write_bytes(end, 0, extended_end as usize - end as usize);
        self.data = extended_end;
    }

    fn size(&self) -> usize {
        self.data as usize - self.base as usize
    }

    /// Recover the data vector of the pass, consuming `self`.
    unsafe fn into_vec(mut self) -> (Vec<u8>, P) {
        self.invalidate()
    }

    /// Make pass contents invalid, return the contained data.
    ///
    /// Any following access to the pass will result in a crash
    /// for accessing address 0.
    pub unsafe fn invalidate(&mut self) -> (Vec<u8>, P) {
        let size = self.size();
        assert!(
            size <= self.capacity,
            "Size of RawPass ({}) exceeds capacity ({})",
            size,
            self.capacity
        );
        let vec = Vec::from_raw_parts(self.base, size, self.capacity);
        self.data = ptr::null_mut();
        self.base = ptr::null_mut();
        self.capacity = 0;
        (vec, self.parent)
    }

    unsafe fn ensure_extra_size(&mut self, extra_size: usize) {
        let size = self.size();
        if size + extra_size > self.capacity {
            let mut vec = Vec::from_raw_parts(self.base, size, self.capacity);
            vec.reserve(extra_size);
            //let (data, size, capacity) = vec.into_raw_parts(); //TODO: when stable
            self.data = vec.as_mut_ptr().add(vec.len());
            self.base = vec.as_mut_ptr();
            self.capacity = vec.capacity();
            mem::forget(vec);
        }
    }

    #[inline]
    pub(crate) unsafe fn encode<C: peek_poke::Poke>(&mut self, command: &C) {
        self.ensure_extra_size(C::max_size());
        self.data = command.poke_into(self.data);
    }

    #[inline]
    pub(crate) unsafe fn encode_slice<T: Copy>(&mut self, data: &[T]) {
        let align_offset = self.data.align_offset(mem::align_of::<T>());
        let extra = align_offset + mem::size_of::<T>() * data.len();
        self.ensure_extra_size(extra);
        slice::from_raw_parts_mut(self.data.add(align_offset) as *mut T, data.len())
            .copy_from_slice(data);
        self.data = self.data.add(extra);
    }
}

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

#[repr(C)]
#[derive(PeekPoke)]
struct PassComponent<T> {
    load_op: wgt::LoadOp,
    store_op: wgt::StoreOp,
    clear_value: T,
    read_only: bool,
}

// required for PeekPoke
impl<T: Default> Default for PassComponent<T> {
    fn default() -> Self {
        PassComponent {
            load_op: wgt::LoadOp::Clear,
            store_op: wgt::StoreOp::Clear,
            clear_value: T::default(),
            read_only: false,
        }
    }
}

#[repr(C)]
#[derive(Default, PeekPoke)]
struct RawRenderPassColorAttachmentDescriptor {
    attachment: u64,
    resolve_target: u64,
    component: PassComponent<wgt::Color>,
}

#[repr(C)]
#[derive(Default, PeekPoke)]
struct RawRenderPassDepthStencilAttachmentDescriptor {
    attachment: u64,
    depth: PassComponent<f32>,
    stencil: PassComponent<u32>,
}

#[repr(C)]
#[derive(Default, PeekPoke)]
struct RawRenderTargets {
    colors: [RawRenderPassColorAttachmentDescriptor; MAX_COLOR_TARGETS],
    depth_stencil: RawRenderPassDepthStencilAttachmentDescriptor,
}

impl<G: GlobalIdentityHandlerFactory> Global<G> {
    pub fn command_encoder_finish<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        _desc: &wgt::CommandBufferDescriptor,
    ) -> id::CommandBufferId {
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
        log::debug!("Command buffer {:?} {:#?}", encoder_id, comb.trackers);
        encoder_id
    }

    pub fn command_encoder_push_debug_group<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        label: &str,
    ) {
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

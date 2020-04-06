/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

mod allocator;
mod bind;
mod compute;
mod render;
mod transfer;

pub(crate) use self::allocator::CommandAllocator;
pub use self::compute::*;
pub use self::render::*;
pub use self::transfer::*;

use crate::{
    device::{all_buffer_stages, all_image_stages, MAX_COLOR_TARGETS},
    hub::{GfxBackend, Global, GlobalIdentityHandlerFactory, Storage, Token},
    id,
    resource::{Buffer, Texture},
    track::TrackerSet,
    Features, LifeGuard, Stored,
};

use peek_poke::PeekPoke;

use std::{marker::PhantomData, mem, ptr, slice, thread::ThreadId};

#[derive(Clone, Copy, Debug, PeekPoke)]
struct PhantomSlice<T>(PhantomData<T>);

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
        assert!(end <= bound);
        (end, slice::from_raw_parts(aligned as *const T, count))
    }
}

#[repr(C)]
pub struct RawPass {
    data: *mut u8,
    base: *mut u8,
    capacity: usize,
    parent: id::CommandEncoderId,
}

impl RawPass {
    fn from_vec<T>(mut vec: Vec<T>, encoder_id: id::CommandEncoderId) -> Self {
        let ptr = vec.as_mut_ptr() as *mut u8;
        let capacity = vec.capacity() * mem::size_of::<T>();
        mem::forget(vec);
        RawPass {
            data: ptr,
            base: ptr,
            capacity,
            parent: encoder_id,
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

    pub unsafe fn into_vec(self) -> (Vec<u8>, id::CommandEncoderId) {
        let size = self.size();
        assert!(size <= self.capacity);
        let vec = Vec::from_raw_parts(self.base, size, self.capacity);
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
    unsafe fn encode<C: peek_poke::Poke>(&mut self, command: &C) {
        self.ensure_extra_size(C::max_size());
        self.data = command.poke_into(self.data);
    }

    #[inline]
    unsafe fn encode_slice<T: Copy>(&mut self, data: &[T]) {
        let align_offset = self.data.align_offset(mem::align_of::<T>());
        let extra = align_offset + mem::size_of::<T>() * data.len();
        self.ensure_extra_size(extra);
        slice::from_raw_parts_mut(self.data.add(align_offset) as *mut T, data.len())
            .copy_from_slice(data);
        self.data = self.data.add(extra);
    }
}

pub struct RenderBundle<B: hal::Backend> {
    _raw: B::CommandBuffer,
}

#[derive(Debug)]
pub struct CommandBuffer<B: hal::Backend> {
    pub(crate) raw: Vec<B::CommandBuffer>,
    is_recording: bool,
    recorded_thread_id: ThreadId,
    pub(crate) device_id: Stored<id::DeviceId>,
    pub(crate) life_guard: LifeGuard,
    pub(crate) trackers: TrackerSet,
    pub(crate) used_swap_chain: Option<(Stored<id::SwapChainId>, B::Framebuffer)>,
    pub(crate) features: Features,
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
}

// required for PeekPoke
impl<T: Default> Default for PassComponent<T> {
    fn default() -> Self {
        PassComponent {
            load_op: wgt::LoadOp::Clear,
            store_op: wgt::StoreOp::Clear,
            clear_value: T::default(),
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
        assert!(comb.is_recording);
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
}

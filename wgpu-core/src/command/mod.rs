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
    device::{
        MAX_COLOR_TARGETS,
        all_buffer_stages,
        all_image_stages,
    },
    hub::{GfxBackend, Global, Storage, Token},
    id,
    resource::{Buffer, Texture},
    track::TrackerSet,
    Features,
    LifeGuard,
    Stored,
};

use std::{
    marker::PhantomData,
    mem,
    ptr,
    slice,
    thread::ThreadId,
};


#[derive(Clone, Copy, Debug, peek_poke::PeekCopy, peek_poke::Poke)]
struct PhantomSlice<T>(PhantomData<T>);

impl<T> PhantomSlice<T> {
    fn new() -> Self {
        PhantomSlice(PhantomData)
    }

    unsafe fn decode_unaligned<'a>(
        self, pointer: *const u8, count: usize, bound: *const u8
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
    fn from_vec<T>(
        mut vec: Vec<T>,
        encoder_id: id::CommandEncoderId,
    ) -> Self {
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
    unsafe fn finish<C: peek_poke::Poke>(
        &mut self, command: C
    ) {
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

        let buffer_barriers = base
            .buffers
            .merge_replace(&head.buffers)
            .map(|pending| {
                let buf = &buffer_guard[pending.id];
                pending.into_hal(buf)
            });
        let texture_barriers = base
            .textures
            .merge_replace(&head.textures)
            .map(|pending| {
                let tex = &texture_guard[pending.id];
                pending.into_hal(tex)
            });
        base.views.merge_extend(&head.views).unwrap();
        base.bind_groups.merge_extend(&head.bind_groups).unwrap();
        base.samplers.merge_extend(&head.samplers).unwrap();

        let stages = all_buffer_stages() | all_image_stages();
        unsafe {
            raw.pipeline_barrier(
                stages .. stages,
                hal::memory::Dependencies::empty(),
                buffer_barriers.chain(texture_barriers),
            );
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct CommandEncoderDescriptor {
    // MSVC doesn't allow zero-sized structs
    // We can remove this when we actually have a field
    pub todo: u32,
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
pub struct CommandBufferDescriptor {
    pub todo: u32,
}

type RawRenderPassColorAttachmentDescriptor =
    RenderPassColorAttachmentDescriptorBase<id::TextureViewId, id::TextureViewId>;

#[repr(C)]
pub struct RawRenderTargets {
    pub colors: [RawRenderPassColorAttachmentDescriptor; MAX_COLOR_TARGETS],
    pub depth_stencil: RenderPassDepthStencilAttachmentDescriptor,
}

#[repr(C)]
pub struct RawRenderPass {
    raw: RawPass,
    targets: RawRenderTargets,
}

/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer
/// (`RenderPassDescriptor::color_attachments`) is valid for
/// `RenderPassDescriptor::color_attachments_length` elements.
#[no_mangle]
pub unsafe extern "C" fn wgpu_command_encoder_begin_render_pass(
    encoder_id: id::CommandEncoderId,
    desc: &RenderPassDescriptor,
) -> *mut RawRenderPass {
    let mut colors: [RawRenderPassColorAttachmentDescriptor; MAX_COLOR_TARGETS] = mem::zeroed();
    for (color, at) in colors
        .iter_mut()
        .zip(slice::from_raw_parts(desc.color_attachments, desc.color_attachments_length))
    {
        *color = RawRenderPassColorAttachmentDescriptor {
            attachment: at.attachment,
            resolve_target: at.resolve_target.map_or(id::TextureViewId::ERROR, |rt| *rt),
            load_op: at.load_op,
            store_op: at.store_op,
            clear_color: at.clear_color,
        };
    }
    let pass = RawRenderPass {
        raw: RawPass::new_render(encoder_id),
        targets: RawRenderTargets {
            colors,
            depth_stencil: desc.depth_stencil_attachment
                .cloned()
                .unwrap_or_else(|| mem::zeroed()),
        },
    };
    Box::into_raw(Box::new(pass))
}

impl<F> Global<F> {
    pub fn command_encoder_finish<B: GfxBackend>(
        &self,
        encoder_id: id::CommandEncoderId,
        _desc: &CommandBufferDescriptor,
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
            let view_id = swap_chain_guard[sc_id.value].acquired_view_id
                .as_ref()
                .expect("Used swap chain frame has already presented");
            comb.trackers.views.remove(view_id.value);
        }
        log::debug!("Command buffer {:?} {:#?}", encoder_id, comb.trackers);
        encoder_id
    }
}

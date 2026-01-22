use alloc::{boxed::Box, vec::Vec};
use smallvec::SmallVec;

use crate::{
    DeviceError, DynCommandBuffer, DynFence, DynResource, DynSurface, DynSurfaceTexture, Queue,
    SurfaceError,
};

use super::DynResourceExt as _;

pub trait DynQueue: DynResource {
    unsafe fn submit<'a>(
        &self,
        submits: &'a mut [crate::QueueSubmitInfo<
            'a,
            dyn DynCommandBuffer,
            dyn DynFence,
            dyn DynSurfaceTexture,
        >],
    ) -> Result<(), DeviceError>;
    unsafe fn present(
        &self,
        surface: &dyn DynSurface,
        texture: Box<dyn DynSurfaceTexture>,
    ) -> Result<(), SurfaceError>;
    unsafe fn get_timestamp_period(&self) -> f32;
}

impl<Q: Queue + DynResource> DynQueue for Q {
    unsafe fn submit<'a>(
        &self,
        submits: &'a mut [crate::QueueSubmitInfo<
            'a,
            dyn DynCommandBuffer,
            dyn DynFence,
            dyn DynSurfaceTexture,
        >],
    ) -> Result<(), DeviceError> {
        if submits.is_empty() {
            return Ok(());
        }

        struct SubmitContext {
            cb_count: usize,
            st_count: usize,
            signal_count: usize,
            wait_count: usize,
        }
        let mut command_buffers =
            SmallVec::<[&<<Q as Queue>::A as crate::Api>::CommandBuffer; 1]>::new();
        let mut surface_textures =
            SmallVec::<[&<<Q as Queue>::A as crate::Api>::SurfaceTexture; 1]>::new();
        let mut fences = Vec::<(&mut <Q::A as crate::Api>::Fence, u64)>::new();

        let mut contexts = Vec::new();
        for submit in submits.iter_mut() {
            contexts.push(SubmitContext {
                cb_count: submit.command_buffers.len(),
                st_count: submit.surface_textures.len(),
                signal_count: submit.signal_fences.len(),
                wait_count: submit.wait_fences.len(),
            });
            for cb in submit.command_buffers {
                command_buffers.push((**cb).expect_downcast_ref());
            }
            for st in submit.surface_textures {
                command_buffers.push((**st).expect_downcast_ref());
            }
            for fence in submit.signal_fences.iter_mut() {
                fences.push(((*fence.0).expect_downcast_mut(), fence.1));
            }
            for fence in submit.wait_fences.iter_mut() {
                fences.push(((*fence.0).expect_downcast_mut(), fence.1));
            }
        }

        let mut current_cb_slice: &[&<Q::A as crate::Api>::CommandBuffer] = &command_buffers;
        let mut current_texture_slice: &[&<Q::A as crate::Api>::SurfaceTexture] =
            &mut surface_textures;
        // Impossible to make the borrow checker happy with smallvec here without pointers & unsafe
        // Thats because of following loops, mutable references to fences everywhere, etc
        let mut current_fence_slice: &mut [(&mut <Q::A as crate::Api>::Fence, u64)] = &mut fences;
        let mut out_submits = Vec::new();

        for ctx in contexts {
            let (cbs, new_current_cb_slice) = current_cb_slice.split_at(ctx.cb_count);
            current_cb_slice = new_current_cb_slice;
            let (sts, new_current_texture_slice) = current_texture_slice.split_at(ctx.st_count);
            current_texture_slice = new_current_texture_slice;
            let (signal_fences, new_current_fence_slice) =
                current_fence_slice.split_at_mut(ctx.signal_count);
            let (wait_fences, new_current_fence_slice) =
                new_current_fence_slice.split_at_mut(ctx.wait_count);
            current_fence_slice = new_current_fence_slice;

            out_submits.push(crate::QueueSubmitInfo {
                command_buffers: cbs,
                surface_textures: sts,
                signal_fences,
                wait_fences,
            })
        }

        unsafe { Q::submit(self, &mut out_submits) }
    }

    unsafe fn present(
        &self,
        surface: &dyn DynSurface,
        texture: Box<dyn DynSurfaceTexture>,
    ) -> Result<(), SurfaceError> {
        let surface = surface.expect_downcast_ref();
        unsafe { Q::present(self, surface, texture.unbox()) }
    }

    unsafe fn get_timestamp_period(&self) -> f32 {
        unsafe { Q::get_timestamp_period(self) }
    }
}

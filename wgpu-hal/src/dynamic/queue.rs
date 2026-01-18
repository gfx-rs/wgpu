use core::marker::PhantomData;

use alloc::{boxed::Box, vec::Vec};

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
        let max_cb_count = submits
            .iter()
            .map(|a| a.command_buffers.iter().len())
            .max()
            .unwrap();
        let max_st_count = submits
            .iter()
            .map(|a| a.surface_textures.iter().len())
            .max()
            .unwrap();

        unsafe {
            Q::submit(
                self,
                TypedSubmitIterator::<'_, Q::A, _> {
                    submits: submits.iter_mut(),
                    command_buffers: Vec::with_capacity(max_cb_count),
                    surface_textures: Vec::with_capacity(max_st_count),
                    _p: Default::default(),
                },
            )
        }
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

struct TypedSubmitIterator<'a, A: crate::Api, I>
where
    I: Iterator<
            Item = &'a mut crate::QueueSubmitInfo<
                'a,
                dyn DynCommandBuffer,
                dyn DynFence,
                dyn DynSurfaceTexture,
            >,
        > + ExactSizeIterator,
{
    submits: I,
    command_buffers: Vec<&'a A::CommandBuffer>,
    surface_textures: Vec<&'a A::SurfaceTexture>,
    _p: PhantomData<A>,
}
impl<'a, A: crate::Api, I> crate::SubmitIterator<A::CommandBuffer, A::Fence, A::SurfaceTexture>
    for TypedSubmitIterator<'a, A, I>
where
    I: ExactSizeIterator
        + Iterator<
            Item = &'a mut crate::QueueSubmitInfo<
                'a,
                dyn DynCommandBuffer,
                dyn DynFence,
                dyn DynSurfaceTexture,
            >,
        >,
{
    fn len(&self) -> usize {
        self.submits.len()
    }
    fn next<'b>(
        &'b mut self,
    ) -> Option<crate::QueueSubmitInfo<'b, A::CommandBuffer, A::Fence, A::SurfaceTexture>> {
        if let Some(submit) = self.submits.next() {
            self.command_buffers.clear();
            self.command_buffers.reserve(submit.command_buffers.len());
            for cb in submit.command_buffers {
                self.command_buffers.push((**cb).expect_downcast_ref());
            }
            self.surface_textures.clear();
            for st in submit.surface_textures {
                self.surface_textures.push((**st).expect_downcast_ref());
            }
            let signal_fence = submit
                .signal_fence
                .as_mut()
                .map(|a| (a.0.expect_downcast_mut(), a.1));
            let wait_fence = submit
                .wait_fence
                .as_mut()
                .map(|a| (a.0.expect_downcast_mut(), a.1));
            Some(crate::QueueSubmitInfo {
                command_buffers: &self.command_buffers,
                surface_textures: &self.surface_textures,
                signal_fence,
                wait_fence,
            })
        } else {
            None
        }
    }
}

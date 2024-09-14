use std::{marker::PhantomData, ops::Range, sync::Arc, thread};

use crate::context::DynContext;
use crate::*;

/// Encodes a series of GPU operations.
///
/// A command encoder can record [`RenderPass`]es, [`ComputePass`]es,
/// and transfer operations between driver-managed resources like [`Buffer`]s and [`Texture`]s.
///
/// When finished recording, call [`CommandEncoder::finish`] to obtain a [`CommandBuffer`] which may
/// be submitted for execution.
///
/// Corresponds to [WebGPU `GPUCommandEncoder`](https://gpuweb.github.io/gpuweb/#command-encoder).
#[derive(Debug)]
pub struct CommandEncoder {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(CommandEncoder: Send, Sync);

impl Drop for CommandEncoder {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.command_encoder_drop(self.data.as_ref());
        }
    }
}

/// Describes a [`CommandEncoder`].
///
/// For use with [`Device::create_command_encoder`].
///
/// Corresponds to [WebGPU `GPUCommandEncoderDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpucommandencoderdescriptor).
pub type CommandEncoderDescriptor<'a> = wgt::CommandEncoderDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(CommandEncoderDescriptor<'_>: Send, Sync);

pub use wgt::ImageCopyBuffer as ImageCopyBufferBase;
/// View of a buffer which can be used to copy to/from a texture.
///
/// Corresponds to [WebGPU `GPUImageCopyBuffer`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopybuffer).
pub type ImageCopyBuffer<'a> = ImageCopyBufferBase<&'a Buffer>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ImageCopyBuffer<'_>: Send, Sync);

pub use wgt::ImageCopyTexture as ImageCopyTextureBase;
/// View of a texture which can be used to copy to/from a buffer/texture.
///
/// Corresponds to [WebGPU `GPUImageCopyTexture`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexture).
pub type ImageCopyTexture<'a> = ImageCopyTextureBase<&'a Texture>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ImageCopyTexture<'_>: Send, Sync);

pub use wgt::ImageCopyTextureTagged as ImageCopyTextureTaggedBase;
/// View of a texture which can be used to copy to a texture, including
/// color space and alpha premultiplication information.
///
/// Corresponds to [WebGPU `GPUImageCopyTextureTagged`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpuimagecopytexturetagged).
pub type ImageCopyTextureTagged<'a> = ImageCopyTextureTaggedBase<&'a Texture>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ImageCopyTexture<'_>: Send, Sync);

impl CommandEncoder {
    /// Finishes recording and returns a [`CommandBuffer`] that can be submitted for execution.
    pub fn finish(mut self) -> CommandBuffer {
        let data = DynContext::command_encoder_finish(&*self.context, self.data.as_mut());
        CommandBuffer {
            context: Arc::clone(&self.context),
            data: Some(data),
        }
    }

    /// Begins recording of a render pass.
    ///
    /// This function returns a [`RenderPass`] object which records a single render pass.
    ///
    /// As long as the returned  [`RenderPass`] has not ended,
    /// any mutating operation on this command encoder causes an error and invalidates it.
    /// Note that the `'encoder` lifetime relationship protects against this,
    /// but it is possible to opt out of it by calling [`RenderPass::forget_lifetime`].
    /// This can be useful for runtime handling of the encoder->pass
    /// dependency e.g. when pass and encoder are stored in the same data structure.
    pub fn begin_render_pass<'encoder>(
        &'encoder mut self,
        desc: &RenderPassDescriptor<'_>,
    ) -> RenderPass<'encoder> {
        let data =
            DynContext::command_encoder_begin_render_pass(&*self.context, self.data.as_ref(), desc);
        RenderPass {
            inner: RenderPassInner {
                data,
                context: self.context.clone(),
            },
            encoder_guard: PhantomData,
        }
    }

    /// Begins recording of a compute pass.
    ///
    /// This function returns a [`ComputePass`] object which records a single compute pass.
    ///
    /// As long as the returned  [`ComputePass`] has not ended,
    /// any mutating operation on this command encoder causes an error and invalidates it.
    /// Note that the `'encoder` lifetime relationship protects against this,
    /// but it is possible to opt out of it by calling [`ComputePass::forget_lifetime`].
    /// This can be useful for runtime handling of the encoder->pass
    /// dependency e.g. when pass and encoder are stored in the same data structure.
    pub fn begin_compute_pass<'encoder>(
        &'encoder mut self,
        desc: &ComputePassDescriptor<'_>,
    ) -> ComputePass<'encoder> {
        let data = DynContext::command_encoder_begin_compute_pass(
            &*self.context,
            self.data.as_ref(),
            desc,
        );
        ComputePass {
            inner: ComputePassInner {
                data,
                context: self.context.clone(),
            },
            encoder_guard: PhantomData,
        }
    }

    /// Copy data from one buffer to another.
    ///
    /// # Panics
    ///
    /// - Buffer offsets or copy size not a multiple of [`COPY_BUFFER_ALIGNMENT`].
    /// - Copy would overrun buffer.
    /// - Copy within the same buffer.
    pub fn copy_buffer_to_buffer(
        &mut self,
        source: &Buffer,
        source_offset: BufferAddress,
        destination: &Buffer,
        destination_offset: BufferAddress,
        copy_size: BufferAddress,
    ) {
        DynContext::command_encoder_copy_buffer_to_buffer(
            &*self.context,
            self.data.as_ref(),
            source.data.as_ref(),
            source_offset,
            destination.data.as_ref(),
            destination_offset,
            copy_size,
        );
    }

    /// Copy data from a buffer to a texture.
    pub fn copy_buffer_to_texture(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    ) {
        DynContext::command_encoder_copy_buffer_to_texture(
            &*self.context,
            self.data.as_ref(),
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from a texture to a buffer.
    pub fn copy_texture_to_buffer(
        &mut self,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyBuffer<'_>,
        copy_size: Extent3d,
    ) {
        DynContext::command_encoder_copy_texture_to_buffer(
            &*self.context,
            self.data.as_ref(),
            source,
            destination,
            copy_size,
        );
    }

    /// Copy data from one texture to another.
    ///
    /// # Panics
    ///
    /// - Textures are not the same type
    /// - If a depth texture, or a multisampled texture, the entire texture must be copied
    /// - Copy would overrun either texture
    pub fn copy_texture_to_texture(
        &mut self,
        source: ImageCopyTexture<'_>,
        destination: ImageCopyTexture<'_>,
        copy_size: Extent3d,
    ) {
        DynContext::command_encoder_copy_texture_to_texture(
            &*self.context,
            self.data.as_ref(),
            source,
            destination,
            copy_size,
        );
    }

    /// Clears texture to zero.
    ///
    /// Note that unlike with clear_buffer, `COPY_DST` usage is not required.
    ///
    /// # Implementation notes
    ///
    /// - implemented either via buffer copies and render/depth target clear, path depends on texture usages
    /// - behaves like texture zero init, but is performed immediately (clearing is *not* delayed via marking it as uninitialized)
    ///
    /// # Panics
    ///
    /// - `CLEAR_TEXTURE` extension not enabled
    /// - Range is out of bounds
    pub fn clear_texture(&mut self, texture: &Texture, subresource_range: &ImageSubresourceRange) {
        DynContext::command_encoder_clear_texture(
            &*self.context,
            self.data.as_ref(),
            texture.data.as_ref(),
            subresource_range,
        );
    }

    /// Clears buffer to zero.
    ///
    /// # Panics
    ///
    /// - Buffer does not have `COPY_DST` usage.
    /// - Range is out of bounds
    pub fn clear_buffer(
        &mut self,
        buffer: &Buffer,
        offset: BufferAddress,
        size: Option<BufferAddress>,
    ) {
        DynContext::command_encoder_clear_buffer(
            &*self.context,
            self.data.as_ref(),
            buffer.data.as_ref(),
            offset,
            size,
        );
    }

    /// Inserts debug marker.
    pub fn insert_debug_marker(&mut self, label: &str) {
        DynContext::command_encoder_insert_debug_marker(&*self.context, self.data.as_ref(), label);
    }

    /// Start record commands and group it into debug marker group.
    pub fn push_debug_group(&mut self, label: &str) {
        DynContext::command_encoder_push_debug_group(&*self.context, self.data.as_ref(), label);
    }

    /// Stops command recording and creates debug group.
    pub fn pop_debug_group(&mut self) {
        DynContext::command_encoder_pop_debug_group(&*self.context, self.data.as_ref());
    }

    /// Resolves a query set, writing the results into the supplied destination buffer.
    ///
    /// Occlusion and timestamp queries are 8 bytes each (see [`crate::QUERY_SIZE`]). For pipeline statistics queries,
    /// see [`PipelineStatisticsTypes`] for more information.
    pub fn resolve_query_set(
        &mut self,
        query_set: &QuerySet,
        query_range: Range<u32>,
        destination: &Buffer,
        destination_offset: BufferAddress,
    ) {
        DynContext::command_encoder_resolve_query_set(
            &*self.context,
            self.data.as_ref(),
            query_set.data.as_ref(),
            query_range.start,
            query_range.end - query_range.start,
            destination.data.as_ref(),
            destination_offset,
        )
    }

    /// Returns the inner hal CommandEncoder using a callback. The hal command encoder will be `None` if the
    /// backend type argument does not match with this wgpu CommandEncoder
    ///
    /// This method will start the wgpu_core level command recording.
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal CommandEncoder must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal_mut<
        A: wgc::hal_api::HalApi,
        F: FnOnce(Option<&mut A::CommandEncoder>) -> R,
        R,
    >(
        &mut self,
        hal_command_encoder_callback: F,
    ) -> Option<R> {
        self.context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
            .map(|ctx| unsafe {
                ctx.command_encoder_as_hal_mut::<A, F, R>(
                    crate::context::downcast_ref(self.data.as_ref()),
                    hal_command_encoder_callback,
                )
            })
    }
}

/// [`Features::TIMESTAMP_QUERY_INSIDE_ENCODERS`] must be enabled on the device in order to call these functions.
impl CommandEncoder {
    /// Issue a timestamp command at this point in the queue.
    /// The timestamp will be written to the specified query set, at the specified index.
    ///
    /// Must be multiplied by [`Queue::get_timestamp_period`] to get
    /// the value in nanoseconds. Absolute values have no meaning,
    /// but timestamps can be subtracted to get the time it takes
    /// for a string of operations to complete.
    ///
    /// Attention: Since commands within a command recorder may be reordered,
    /// there is no strict guarantee that timestamps are taken after all commands
    /// recorded so far and all before all commands recorded after.
    /// This may depend both on the backend and the driver.
    pub fn write_timestamp(&mut self, query_set: &QuerySet, query_index: u32) {
        DynContext::command_encoder_write_timestamp(
            &*self.context,
            self.data.as_mut(),
            query_set.data.as_ref(),
            query_index,
        )
    }
}

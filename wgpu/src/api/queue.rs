use std::ops::{Deref, DerefMut};

use crate::*;

/// Handle to a command queue on a device.
///
/// A `Queue` executes recorded [`CommandBuffer`] objects and provides convenience methods
/// for writing to [buffers](Queue::write_buffer) and [textures](Queue::write_texture).
/// It can be created along with a [`Device`] by calling [`Adapter::request_device`].
///
/// Corresponds to [WebGPU `GPUQueue`](https://gpuweb.github.io/gpuweb/#gpu-queue).
#[derive(Debug)]
pub struct Queue {
    pub(crate) inner: dispatch::DispatchQueue,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Queue: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(Queue => .inner);

/// Identifier for a particular call to [`Queue::submit`]. Can be used
/// as part of an argument to [`Device::poll`] to block for a particular
/// submission to finish.
///
/// This type is unique to the Rust API of `wgpu`.
/// There is no analogue in the WebGPU specification.
#[derive(Debug, Clone)]
pub struct SubmissionIndex {
    #[cfg_attr(not(native), allow(dead_code))]
    pub(crate) index: u64,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(SubmissionIndex: Send, Sync);

pub use wgt::Maintain as MaintainBase;
/// Passed to [`Device::poll`] to control how and if it should block.
pub type Maintain = wgt::Maintain<SubmissionIndex>;
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Maintain: Send, Sync);

/// A write-only view into a staging buffer.
///
/// Reading into this buffer won't yield the contents of the buffer from the
/// GPU and is likely to be slow. Because of this, although [`AsMut`] is
/// implemented for this type, [`AsRef`] is not.
pub struct QueueWriteBufferView<'a> {
    queue: &'a Queue,
    buffer: &'a Buffer,
    offset: BufferAddress,
    inner: dispatch::DispatchQueueWriteBuffer,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(QueueWriteBufferView<'_>: Send, Sync);

impl Deref for QueueWriteBufferView<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        log::warn!("Reading from a QueueWriteBufferView won't yield the contents of the buffer and may be slow.");
        self.inner.slice()
    }
}

impl DerefMut for QueueWriteBufferView<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.slice_mut()
    }
}

impl AsMut<[u8]> for QueueWriteBufferView<'_> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.inner.slice_mut()
    }
}

impl Drop for QueueWriteBufferView<'_> {
    fn drop(&mut self) {
        self.queue
            .inner
            .write_staging_buffer(&self.buffer.inner, self.offset, &self.inner);
    }
}

impl Queue {
    /// Schedule a data write into `buffer` starting at `offset`.
    ///
    /// This method fails if `data` overruns the size of `buffer` starting at `offset`.
    ///
    /// This does *not* submit the transfer to the GPU immediately. Calls to
    /// `write_buffer` begin execution only on the next call to
    /// [`Queue::submit`]. To get a set of scheduled transfers started
    /// immediately, it's fine to call `submit` with no command buffers at all:
    ///
    /// ```no_run
    /// # let queue: wgpu::Queue = todo!();
    /// queue.submit([]);
    /// ```
    ///
    /// However, `data` will be immediately copied into staging memory, so the
    /// caller may discard it any time after this call completes.
    ///
    /// If possible, consider using [`Queue::write_buffer_with`] instead. That
    /// method avoids an intermediate copy and is often able to transfer data
    /// more efficiently than this one.
    pub fn write_buffer(&self, buffer: &Buffer, offset: BufferAddress, data: &[u8]) {
        self.inner.write_buffer(&buffer.inner, offset, data);
    }

    /// Write to a buffer via a directly mapped staging buffer.
    ///
    /// Return a [`QueueWriteBufferView`] which, when dropped, schedules a copy
    /// of its contents into `buffer` at `offset`. The returned view
    /// dereferences to a `size`-byte long `&mut [u8]`, in which you should
    /// store the data you would like written to `buffer`.
    ///
    /// This method may perform transfers faster than [`Queue::write_buffer`],
    /// because the returned [`QueueWriteBufferView`] is actually the staging
    /// buffer for the write, mapped into the caller's address space. Writing
    /// your data directly into this staging buffer avoids the temporary
    /// CPU-side buffer needed by `write_buffer`.
    ///
    /// Reading from the returned view is slow, and will not yield the current
    /// contents of `buffer`.
    ///
    /// Note that dropping the [`QueueWriteBufferView`] does *not* submit the
    /// transfer to the GPU immediately. The transfer begins only on the next
    /// call to [`Queue::submit`] after the view is dropped. To get a set of
    /// scheduled transfers started immediately, it's fine to call `submit` with
    /// no command buffers at all:
    ///
    /// ```no_run
    /// # let queue: wgpu::Queue = todo!();
    /// queue.submit([]);
    /// ```
    ///
    /// This method fails if `size` is greater than the size of `buffer` starting at `offset`.
    #[must_use]
    pub fn write_buffer_with<'a>(
        &'a self,
        buffer: &'a Buffer,
        offset: BufferAddress,
        size: BufferSize,
    ) -> Option<QueueWriteBufferView<'a>> {
        profiling::scope!("Queue::write_buffer_with");
        self.inner
            .validate_write_buffer(&buffer.inner, offset, size)?;
        let staging_buffer = self.inner.create_staging_buffer(size)?;
        Some(QueueWriteBufferView {
            queue: self,
            buffer,
            offset,
            inner: staging_buffer,
        })
    }

    /// Schedule a write of some data into a texture.
    ///
    /// * `data` contains the texels to be written, which must be in
    ///   [the same format as the texture](TextureFormat).
    /// * `data_layout` describes the memory layout of `data`, which does not necessarily
    ///   have to have tightly packed rows.
    /// * `texture` specifies the texture to write into, and the location within the
    ///   texture (coordinate offset, mip level) that will be overwritten.
    /// * `size` is the size, in texels, of the region to be written.
    ///
    /// This method fails if `size` overruns the size of `texture`, or if `data` is too short.
    ///
    /// This does *not* submit the transfer to the GPU immediately. Calls to
    /// `write_texture` begin execution only on the next call to
    /// [`Queue::submit`]. To get a set of scheduled transfers started
    /// immediately, it's fine to call `submit` with no command buffers at all:
    ///
    /// ```no_run
    /// # let queue: wgpu::Queue = todo!();
    /// queue.submit([]);
    /// ```
    ///
    /// However, `data` will be immediately copied into staging memory, so the
    /// caller may discard it any time after this call completes.
    pub fn write_texture(
        &self,
        texture: TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: TexelCopyBufferLayout,
        size: Extent3d,
    ) {
        self.inner.write_texture(texture, data, data_layout, size);
    }

    /// Schedule a copy of data from `image` into `texture`.
    #[cfg(any(webgpu, webgl))]
    pub fn copy_external_image_to_texture(
        &self,
        source: &wgt::CopyExternalImageSourceInfo,
        dest: wgt::CopyExternalImageDestInfo<&api::Texture>,
        size: Extent3d,
    ) {
        self.inner
            .copy_external_image_to_texture(source, dest, size);
    }

    /// Submits a series of finished command buffers for execution.
    pub fn submit<I: IntoIterator<Item = CommandBuffer>>(
        &self,
        command_buffers: I,
    ) -> SubmissionIndex {
        let mut command_buffers = command_buffers
            .into_iter()
            .map(|mut comb| comb.inner.take().unwrap());

        let index = self.inner.submit(&mut command_buffers);

        SubmissionIndex { index }
    }

    /// Gets the amount of nanoseconds each tick of a timestamp query represents.
    ///
    /// Returns zero if timestamp queries are unsupported.
    ///
    /// Timestamp values are represented in nanosecond values on WebGPU, see `<https://gpuweb.github.io/gpuweb/#timestamp>`
    /// Therefore, this is always 1.0 on the web, but on wgpu-core a manual conversion is required.
    pub fn get_timestamp_period(&self) -> f32 {
        self.inner.get_timestamp_period()
    }

    /// Registers a callback when the previous call to submit finishes running on the gpu. This callback
    /// being called implies that all mapped buffer callbacks which were registered before this call will
    /// have been called.
    ///
    /// For the callback to complete, either `queue.submit(..)`, `instance.poll_all(..)`, or `device.poll(..)`
    /// must be called elsewhere in the runtime, possibly integrated into an event loop or run on a separate thread.
    ///
    /// The callback will be called on the thread that first calls the above functions after the gpu work
    /// has completed. There are no restrictions on the code you can run in the callback, however on native the
    /// call to the function will not complete until the callback returns, so prefer keeping callbacks short
    /// and used to set flags, send messages, etc.
    pub fn on_submitted_work_done(&self, callback: impl FnOnce() + Send + 'static) {
        self.inner.on_submitted_work_done(Box::new(callback));
    }
}

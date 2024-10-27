use std::{
    error, fmt,
    ops::{Bound, Deref, DerefMut, Range, RangeBounds},
    sync::Arc,
    thread,
};

use parking_lot::Mutex;

use crate::context::DynContext;
use crate::*;

/// Handle to a GPU-accessible buffer.
///
/// Created with [`Device::create_buffer`] or
/// [`DeviceExt::create_buffer_init`](util::DeviceExt::create_buffer_init).
///
/// Corresponds to [WebGPU `GPUBuffer`](https://gpuweb.github.io/gpuweb/#buffer-interface).
///
/// A `Buffer`'s bytes have "interior mutability": functions like
/// [`Queue::write_buffer`] or [mapping] a buffer for writing only require a
/// `&Buffer`, not a `&mut Buffer`, even though they modify its contents. `wgpu`
/// prevents simultaneous reads and writes of buffer contents using run-time
/// checks.
///
/// [mapping]: Buffer#mapping-buffers
///
/// # Mapping buffers
///
/// If a `Buffer` is created with the appropriate [`usage`], it can be *mapped*:
/// you can make its contents accessible to the CPU as an ordinary `&[u8]` or
/// `&mut [u8]` slice of bytes. Buffers created with the
/// [`mapped_at_creation`][mac] flag set are also mapped initially.
///
/// Depending on the hardware, the buffer could be memory shared between CPU and
/// GPU, so that the CPU has direct access to the same bytes the GPU will
/// consult; or it may be ordinary CPU memory, whose contents the system must
/// copy to/from the GPU as needed. This crate's API is designed to work the
/// same way in either case: at any given time, a buffer is either mapped and
/// available to the CPU, or unmapped and ready for use by the GPU, but never
/// both. This makes it impossible for either side to observe changes by the
/// other immediately, and any necessary transfers can be carried out when the
/// buffer transitions from one state to the other.
///
/// There are two ways to map a buffer:
///
/// - If [`BufferDescriptor::mapped_at_creation`] is `true`, then the entire
///   buffer is mapped when it is created. This is the easiest way to initialize
///   a new buffer. You can set `mapped_at_creation` on any kind of buffer,
///   regardless of its [`usage`] flags.
///
/// - If the buffer's [`usage`] includes the [`MAP_READ`] or [`MAP_WRITE`]
///   flags, then you can call `buffer.slice(range).map_async(mode, callback)`
///   to map the portion of `buffer` given by `range`. This waits for the GPU to
///   finish using the buffer, and invokes `callback` as soon as the buffer is
///   safe for the CPU to access.
///
/// Once a buffer is mapped:
///
/// - You can call `buffer.slice(range).get_mapped_range()` to obtain a
///   [`BufferView`], which dereferences to a `&[u8]` that you can use to read
///   the buffer's contents.
///
/// - Or, you can call `buffer.slice(range).get_mapped_range_mut()` to obtain a
///   [`BufferViewMut`], which dereferences to a `&mut [u8]` that you can use to
///   read and write the buffer's contents.
///
/// The given `range` must fall within the mapped portion of the buffer. If you
/// attempt to access overlapping ranges, even for shared access only, these
/// methods panic.
///
/// While a buffer is mapped, you may not submit any commands to the GPU that
/// access it. You may record command buffers that use the buffer, but if you
/// submit them while the buffer is mapped, submission will panic.
///
/// When you are done using the buffer on the CPU, you must call
/// [`Buffer::unmap`] to make it available for use by the GPU again. All
/// [`BufferView`] and [`BufferViewMut`] views referring to the buffer must be
/// dropped before you unmap it; otherwise, [`Buffer::unmap`] will panic.
///
/// # Example
///
/// If `buffer` was created with [`BufferUsages::MAP_WRITE`], we could fill it
/// with `f32` values like this:
///
/// ```no_run
/// # mod bytemuck {
/// #     pub fn cast_slice_mut(bytes: &mut [u8]) -> &mut [f32] { todo!() }
/// # }
/// # let device: wgpu::Device = todo!();
/// # let buffer: wgpu::Buffer = todo!();
/// let buffer = std::sync::Arc::new(buffer);
/// let capturable = buffer.clone();
/// buffer.slice(..).map_async(wgpu::MapMode::Write, move |result| {
///     if result.is_ok() {
///         let mut view = capturable.slice(..).get_mapped_range_mut();
///         let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
///         floats.fill(42.0);
///         drop(view);
///         capturable.unmap();
///     }
/// });
/// ```
///
/// This code takes the following steps:
///
/// - First, it moves `buffer` into an [`Arc`], and makes a clone for capture by
///   the callback passed to [`map_async`]. Since a [`map_async`] callback may be
///   invoked from another thread, interaction between the callback and the
///   thread calling [`map_async`] generally requires some sort of shared heap
///   data like this. In real code, the [`Arc`] would probably own some larger
///   structure that itself owns `buffer`.
///
/// - Then, it calls [`Buffer::slice`] to make a [`BufferSlice`] referring to
///   the buffer's entire contents.
///
/// - Next, it calls [`BufferSlice::map_async`] to request that the bytes to
///   which the slice refers be made accessible to the CPU ("mapped"). This may
///   entail waiting for previously enqueued operations on `buffer` to finish.
///   Although [`map_async`] itself always returns immediately, it saves the
///   callback function to be invoked later.
///
/// - When some later call to [`Device::poll`] or [`Instance::poll_all`] (not
///   shown in this example) determines that the buffer is mapped and ready for
///   the CPU to use, it invokes the callback function.
///
/// - The callback function calls [`Buffer::slice`] and then
///   [`BufferSlice::get_mapped_range_mut`] to obtain a [`BufferViewMut`], which
///   dereferences to a `&mut [u8]` slice referring to the buffer's bytes.
///
/// - It then uses the [`bytemuck`] crate to turn the `&mut [u8]` into a `&mut
///   [f32]`, and calls the slice [`fill`] method to fill the buffer with a
///   useful value.
///
/// - Finally, the callback drops the view and calls [`Buffer::unmap`] to unmap
///   the buffer. In real code, the callback would also need to do some sort of
///   synchronization to let the rest of the program know that it has completed
///   its work.
///
/// If using [`map_async`] directly is awkward, you may find it more convenient to
/// use [`Queue::write_buffer`] and [`util::DownloadBuffer::read_buffer`].
/// However, those each have their own tradeoffs; the asynchronous nature of GPU
/// execution makes it hard to avoid friction altogether.
///
/// [`Arc`]: std::sync::Arc
/// [`map_async`]: BufferSlice::map_async
/// [`bytemuck`]: https://crates.io/crates/bytemuck
/// [`fill`]: slice::fill
///
/// ## Mapping buffers on the web
///
/// When compiled to WebAssembly and running in a browser content process,
/// `wgpu` implements its API in terms of the browser's WebGPU implementation.
/// In this context, `wgpu` is further isolated from the GPU:
///
/// - Depending on the browser's WebGPU implementation, mapping and unmapping
///   buffers probably entails copies between WebAssembly linear memory and the
///   graphics driver's buffers.
///
/// - All modern web browsers isolate web content in its own sandboxed process,
///   which can only interact with the GPU via interprocess communication (IPC).
///   Although most browsers' IPC systems use shared memory for large data
///   transfers, there will still probably need to be copies into and out of the
///   shared memory buffers.
///
/// All of these copies contribute to the cost of buffer mapping in this
/// configuration.
///
/// [`usage`]: BufferDescriptor::usage
/// [mac]: BufferDescriptor::mapped_at_creation
/// [`MAP_READ`]: BufferUsages::MAP_READ
/// [`MAP_WRITE`]: BufferUsages::MAP_WRITE
#[derive(Debug)]
pub struct Buffer {
    pub(crate) context: Arc<C>,
    pub(crate) data: Box<Data>,
    pub(crate) map_context: Mutex<MapContext>,
    pub(crate) size: wgt::BufferAddress,
    pub(crate) usage: BufferUsages,
    // Todo: missing map_state https://www.w3.org/TR/webgpu/#dom-gpubuffer-mapstate
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(Buffer: Send, Sync);

super::impl_partialeq_eq_hash!(Buffer);

impl Buffer {
    /// Return the binding view of the entire buffer.
    pub fn as_entire_binding(&self) -> BindingResource<'_> {
        BindingResource::Buffer(self.as_entire_buffer_binding())
    }

    /// Return the binding view of the entire buffer.
    pub fn as_entire_buffer_binding(&self) -> BufferBinding<'_> {
        BufferBinding {
            buffer: self,
            offset: 0,
            size: None,
        }
    }

    /// Returns the inner hal Buffer using a callback. The hal buffer will be `None` if the
    /// backend type argument does not match with this wgpu Buffer
    ///
    /// # Safety
    ///
    /// - The raw handle obtained from the hal Buffer must not be manually destroyed
    #[cfg(wgpu_core)]
    pub unsafe fn as_hal<A: wgc::hal_api::HalApi, F: FnOnce(Option<&A::Buffer>) -> R, R>(
        &self,
        hal_buffer_callback: F,
    ) -> R {
        if let Some(ctx) = self
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWgpuCore>()
        {
            unsafe {
                ctx.buffer_as_hal::<A, F, R>(
                    crate::context::downcast_ref(self.data.as_ref()),
                    hal_buffer_callback,
                )
            }
        } else {
            hal_buffer_callback(None)
        }
    }

    /// Return a slice of a [`Buffer`]'s bytes.
    ///
    /// Return a [`BufferSlice`] referring to the portion of `self`'s contents
    /// indicated by `bounds`. Regardless of what sort of data `self` stores,
    /// `bounds` start and end are given in bytes.
    ///
    /// A [`BufferSlice`] can be used to supply vertex and index data, or to map
    /// buffer contents for access from the CPU. See the [`BufferSlice`]
    /// documentation for details.
    ///
    /// The `range` argument can be half or fully unbounded: for example,
    /// `buffer.slice(..)` refers to the entire buffer, and `buffer.slice(n..)`
    /// refers to the portion starting at the `n`th byte and extending to the
    /// end of the buffer.
    pub fn slice<S: RangeBounds<BufferAddress>>(&self, bounds: S) -> BufferSlice<'_> {
        let (offset, size) = range_to_offset_size(bounds);
        check_buffer_bounds(self.size, offset, size);
        BufferSlice {
            buffer: self,
            offset,
            size,
        }
    }

    /// Flushes any pending write operations and unmaps the buffer from host memory.
    pub fn unmap(&self) {
        self.map_context.lock().reset();
        DynContext::buffer_unmap(&*self.context, self.data.as_ref());
    }

    /// Destroy the associated native resources as soon as possible.
    pub fn destroy(&self) {
        DynContext::buffer_destroy(&*self.context, self.data.as_ref());
    }

    /// Returns the length of the buffer allocation in bytes.
    ///
    /// This is always equal to the `size` that was specified when creating the buffer.
    pub fn size(&self) -> BufferAddress {
        self.size
    }

    /// Returns the allowed usages for this `Buffer`.
    ///
    /// This is always equal to the `usage` that was specified when creating the buffer.
    pub fn usage(&self) -> BufferUsages {
        self.usage
    }
}

/// A slice of a [`Buffer`], to be mapped, used for vertex or index data, or the like.
///
/// You can create a `BufferSlice` by calling [`Buffer::slice`]:
///
/// ```no_run
/// # let buffer: wgpu::Buffer = todo!();
/// let slice = buffer.slice(10..20);
/// ```
///
/// This returns a slice referring to the second ten bytes of `buffer`. To get a
/// slice of the entire `Buffer`:
///
/// ```no_run
/// # let buffer: wgpu::Buffer = todo!();
/// let whole_buffer_slice = buffer.slice(..);
/// ```
///
/// You can pass buffer slices to methods like [`RenderPass::set_vertex_buffer`]
/// and [`RenderPass::set_index_buffer`] to indicate which portion of the buffer
/// a draw call should consult.
///
/// To access the slice's contents on the CPU, you must first [map] the buffer,
/// and then call [`BufferSlice::get_mapped_range`] or
/// [`BufferSlice::get_mapped_range_mut`] to obtain a view of the slice's
/// contents. See the documentation on [mapping][map] for more details,
/// including example code.
///
/// Unlike a Rust shared slice `&[T]`, whose existence guarantees that
/// nobody else is modifying the `T` values to which it refers, a
/// [`BufferSlice`] doesn't guarantee that the buffer's contents aren't
/// changing. You can still record and submit commands operating on the
/// buffer while holding a [`BufferSlice`]. A [`BufferSlice`] simply
/// represents a certain range of the buffer's bytes.
///
/// The `BufferSlice` type is unique to the Rust API of `wgpu`. In the WebGPU
/// specification, an offset and size are specified as arguments to each call
/// working with the [`Buffer`], instead.
///
/// [map]: Buffer#mapping-buffers
#[derive(Copy, Clone, Debug)]
pub struct BufferSlice<'a> {
    pub(crate) buffer: &'a Buffer,
    pub(crate) offset: BufferAddress,
    pub(crate) size: Option<BufferSize>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(BufferSlice<'_>: Send, Sync);

impl<'a> BufferSlice<'a> {
    /// Map the buffer. Buffer is ready to map once the callback is called.
    ///
    /// For the callback to complete, either `queue.submit(..)`, `instance.poll_all(..)`, or `device.poll(..)`
    /// must be called elsewhere in the runtime, possibly integrated into an event loop or run on a separate thread.
    ///
    /// The callback will be called on the thread that first calls the above functions after the gpu work
    /// has completed. There are no restrictions on the code you can run in the callback, however on native the
    /// call to the function will not complete until the callback returns, so prefer keeping callbacks short
    /// and used to set flags, send messages, etc.
    pub fn map_async(
        &self,
        mode: MapMode,
        callback: impl FnOnce(Result<(), BufferAsyncError>) + WasmNotSend + 'static,
    ) {
        let mut mc = self.buffer.map_context.lock();
        assert_eq!(mc.initial_range, 0..0, "Buffer is already mapped");
        let end = match self.size {
            Some(s) => self.offset + s.get(),
            None => mc.total_size,
        };
        mc.initial_range = self.offset..end;

        DynContext::buffer_map_async(
            &*self.buffer.context,
            self.buffer.data.as_ref(),
            mode,
            self.offset..end,
            Box::new(callback),
        )
    }

    /// Gain read-only access to the bytes of a [mapped] [`Buffer`].
    ///
    /// Return a [`BufferView`] referring to the buffer range represented by
    /// `self`. See the documentation for [`BufferView`] for details.
    ///
    /// # Panics
    ///
    /// - This panics if the buffer to which `self` refers is not currently
    ///   [mapped].
    ///
    /// - If you try to create overlapping views of a buffer, mutable or
    ///   otherwise, `get_mapped_range` will panic.
    ///
    /// [mapped]: Buffer#mapping-buffers
    pub fn get_mapped_range(&self) -> BufferView<'a> {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = DynContext::buffer_get_mapped_range(
            &*self.buffer.context,
            self.buffer.data.as_ref(),
            self.offset..end,
        );
        BufferView { slice: *self, data }
    }

    /// Synchronously and immediately map a buffer for reading. If the buffer is not immediately mappable
    /// through [`BufferDescriptor::mapped_at_creation`] or [`BufferSlice::map_async`], will fail.
    ///
    /// This is useful when targeting WebGPU and you want to pass mapped data directly to js.
    /// Unlike `get_mapped_range` which unconditionally copies mapped data into the wasm heap,
    /// this function directly hands you the ArrayBuffer that we mapped the data into in js.
    ///
    /// This is only available on WebGPU, on any other backends this will return `None`.
    #[cfg(webgpu)]
    pub fn get_mapped_range_as_array_buffer(&self) -> Option<js_sys::ArrayBuffer> {
        self.buffer
            .context
            .as_any()
            .downcast_ref::<crate::backend::ContextWebGpu>()
            .map(|ctx| {
                let buffer_data = crate::context::downcast_ref(self.buffer.data.as_ref());
                let end = self.buffer.map_context.lock().add(self.offset, self.size);
                ctx.buffer_get_mapped_range_as_array_buffer(buffer_data, self.offset..end)
            })
    }

    /// Gain write access to the bytes of a [mapped] [`Buffer`].
    ///
    /// Return a [`BufferViewMut`] referring to the buffer range represented by
    /// `self`. See the documentation for [`BufferViewMut`] for more details.
    ///
    /// # Panics
    ///
    /// - This panics if the buffer to which `self` refers is not currently
    ///   [mapped].
    ///
    /// - If you try to create overlapping views of a buffer, mutable or
    ///   otherwise, `get_mapped_range_mut` will panic.
    ///
    /// [mapped]: Buffer#mapping-buffers
    pub fn get_mapped_range_mut(&self) -> BufferViewMut<'a> {
        let end = self.buffer.map_context.lock().add(self.offset, self.size);
        let data = DynContext::buffer_get_mapped_range(
            &*self.buffer.context,
            self.buffer.data.as_ref(),
            self.offset..end,
        );
        BufferViewMut {
            slice: *self,
            data,
            readable: self.buffer.usage.contains(BufferUsages::MAP_READ),
        }
    }
}

/// The mapped portion of a buffer, if any, and its outstanding views.
///
/// This ensures that views fall within the mapped range and don't overlap, and
/// also takes care of turning `Option<BufferSize>` sizes into actual buffer
/// offsets.
#[derive(Debug)]
pub(crate) struct MapContext {
    /// The overall size of the buffer.
    ///
    /// This is just a convenient copy of [`Buffer::size`].
    pub(crate) total_size: BufferAddress,

    /// The range of the buffer that is mapped.
    ///
    /// This is `0..0` if the buffer is not mapped. This becomes non-empty when
    /// the buffer is mapped at creation time, and when you call `map_async` on
    /// some [`BufferSlice`] (so technically, it indicates the portion that is
    /// *or has been requested to be* mapped.)
    ///
    /// All [`BufferView`]s and [`BufferViewMut`]s must fall within this range.
    pub(crate) initial_range: Range<BufferAddress>,

    /// The ranges covered by all outstanding [`BufferView`]s and
    /// [`BufferViewMut`]s. These are non-overlapping, and are all contained
    /// within `initial_range`.
    sub_ranges: Vec<Range<BufferAddress>>,
}

impl MapContext {
    pub(crate) fn new(total_size: BufferAddress) -> Self {
        Self {
            total_size,
            initial_range: 0..0,
            sub_ranges: Vec::new(),
        }
    }

    /// Record that the buffer is no longer mapped.
    fn reset(&mut self) {
        self.initial_range = 0..0;

        assert!(
            self.sub_ranges.is_empty(),
            "You cannot unmap a buffer that still has accessible mapped views"
        );
    }

    /// Record that the `size` bytes of the buffer at `offset` are now viewed.
    ///
    /// Return the byte offset within the buffer of the end of the viewed range.
    ///
    /// # Panics
    ///
    /// This panics if the given range overlaps with any existing range.
    fn add(&mut self, offset: BufferAddress, size: Option<BufferSize>) -> BufferAddress {
        let end = match size {
            Some(s) => offset + s.get(),
            None => self.initial_range.end,
        };
        assert!(self.initial_range.start <= offset && end <= self.initial_range.end);
        // This check is essential for avoiding undefined behavior: it is the
        // only thing that ensures that `&mut` references to the buffer's
        // contents don't alias anything else.
        for sub in self.sub_ranges.iter() {
            assert!(
                end <= sub.start || offset >= sub.end,
                "Intersecting map range with {sub:?}"
            );
        }
        self.sub_ranges.push(offset..end);
        end
    }

    /// Record that the `size` bytes of the buffer at `offset` are no longer viewed.
    ///
    /// # Panics
    ///
    /// This panics if the given range does not exactly match one previously
    /// passed to [`add`].
    ///
    /// [`add]`: MapContext::add
    fn remove(&mut self, offset: BufferAddress, size: Option<BufferSize>) {
        let end = match size {
            Some(s) => offset + s.get(),
            None => self.initial_range.end,
        };

        let index = self
            .sub_ranges
            .iter()
            .position(|r| *r == (offset..end))
            .expect("unable to remove range from map context");
        self.sub_ranges.swap_remove(index);
    }
}

/// Describes a [`Buffer`].
///
/// For use with [`Device::create_buffer`].
///
/// Corresponds to [WebGPU `GPUBufferDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpubufferdescriptor).
pub type BufferDescriptor<'a> = wgt::BufferDescriptor<Label<'a>>;
static_assertions::assert_impl_all!(BufferDescriptor<'_>: Send, Sync);

/// Error occurred when trying to async map a buffer.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BufferAsyncError;
static_assertions::assert_impl_all!(BufferAsyncError: Send, Sync);

impl fmt::Display for BufferAsyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Error occurred when trying to async map a buffer")
    }
}

impl error::Error for BufferAsyncError {}

/// Type of buffer mapping.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum MapMode {
    /// Map only for reading
    Read,
    /// Map only for writing
    Write,
}
static_assertions::assert_impl_all!(MapMode: Send, Sync);

/// A read-only view of a mapped buffer's bytes.
///
/// To get a `BufferView`, first [map] the buffer, and then
/// call `buffer.slice(range).get_mapped_range()`.
///
/// `BufferView` dereferences to `&[u8]`, so you can use all the usual Rust
/// slice methods to access the buffer's contents. It also implements
/// `AsRef<[u8]>`, if that's more convenient.
///
/// Before the buffer can be unmapped, all `BufferView`s observing it
/// must be dropped. Otherwise, the call to [`Buffer::unmap`] will panic.
///
/// For example code, see the documentation on [mapping buffers][map].
///
/// [map]: Buffer#mapping-buffers
/// [`map_async`]: BufferSlice::map_async
#[derive(Debug)]
pub struct BufferView<'a> {
    slice: BufferSlice<'a>,
    data: Box<dyn crate::context::BufferMappedRange>,
}

impl std::ops::Deref for BufferView<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.data.slice()
    }
}

impl AsRef<[u8]> for BufferView<'_> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.data.slice()
    }
}

/// A write-only view of a mapped buffer's bytes.
///
/// To get a `BufferViewMut`, first [map] the buffer, and then
/// call `buffer.slice(range).get_mapped_range_mut()`.
///
/// `BufferViewMut` dereferences to `&mut [u8]`, so you can use all the usual
/// Rust slice methods to access the buffer's contents. It also implements
/// `AsMut<[u8]>`, if that's more convenient.
///
/// It is possible to read the buffer using this view, but doing so is not
/// recommended, as it is likely to be slow.
///
/// Before the buffer can be unmapped, all `BufferViewMut`s observing it
/// must be dropped. Otherwise, the call to [`Buffer::unmap`] will panic.
///
/// For example code, see the documentation on [mapping buffers][map].
///
/// [map]: Buffer#mapping-buffers
#[derive(Debug)]
pub struct BufferViewMut<'a> {
    slice: BufferSlice<'a>,
    data: Box<dyn crate::context::BufferMappedRange>,
    readable: bool,
}

impl AsMut<[u8]> for BufferViewMut<'_> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self.data.slice_mut()
    }
}

impl Deref for BufferViewMut<'_> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        if !self.readable {
            log::warn!("Reading from a BufferViewMut is slow and not recommended.");
        }

        self.data.slice()
    }
}

impl DerefMut for BufferViewMut<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.slice_mut()
    }
}

impl Drop for BufferView<'_> {
    fn drop(&mut self) {
        self.slice
            .buffer
            .map_context
            .lock()
            .remove(self.slice.offset, self.slice.size);
    }
}

impl Drop for BufferViewMut<'_> {
    fn drop(&mut self) {
        self.slice
            .buffer
            .map_context
            .lock()
            .remove(self.slice.offset, self.slice.size);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if !thread::panicking() {
            self.context.buffer_drop(self.data.as_ref());
        }
    }
}

fn check_buffer_bounds(
    buffer_size: BufferAddress,
    offset: BufferAddress,
    size: Option<BufferSize>,
) {
    // A slice of length 0 is invalid, so the offset must not be equal to or greater than the buffer size.
    if offset >= buffer_size {
        panic!(
            "slice offset {} is out of range for buffer of size {}",
            offset, buffer_size
        );
    }

    if let Some(size) = size {
        // Detect integer overflow.
        let end = offset.checked_add(size.get());
        if end.map_or(true, |end| end > buffer_size) {
            panic!(
                "slice offset {} size {} is out of range for buffer of size {}",
                offset, size, buffer_size
            );
        }
    }
}

fn range_to_offset_size<S: RangeBounds<BufferAddress>>(
    bounds: S,
) -> (BufferAddress, Option<BufferSize>) {
    let offset = match bounds.start_bound() {
        Bound::Included(&bound) => bound,
        Bound::Excluded(&bound) => bound + 1,
        Bound::Unbounded => 0,
    };
    let size = match bounds.end_bound() {
        Bound::Included(&bound) => Some(bound + 1 - offset),
        Bound::Excluded(&bound) => Some(bound - offset),
        Bound::Unbounded => None,
    }
    .map(|size| BufferSize::new(size).expect("Buffer slices can not be empty"));

    (offset, size)
}

#[cfg(test)]
mod tests {
    use super::{check_buffer_bounds, range_to_offset_size, BufferSize};

    #[test]
    fn range_to_offset_size_works() {
        assert_eq!(range_to_offset_size(0..2), (0, BufferSize::new(2)));
        assert_eq!(range_to_offset_size(2..5), (2, BufferSize::new(3)));
        assert_eq!(range_to_offset_size(..), (0, None));
        assert_eq!(range_to_offset_size(21..), (21, None));
        assert_eq!(range_to_offset_size(0..), (0, None));
        assert_eq!(range_to_offset_size(..21), (0, BufferSize::new(21)));
    }

    #[test]
    #[should_panic]
    fn range_to_offset_size_panics_for_empty_range() {
        range_to_offset_size(123..123);
    }

    #[test]
    #[should_panic]
    fn range_to_offset_size_panics_for_unbounded_empty_range() {
        range_to_offset_size(..0);
    }

    #[test]
    #[should_panic]
    fn check_buffer_bounds_panics_for_offset_at_size() {
        check_buffer_bounds(100, 100, None);
    }

    #[test]
    fn check_buffer_bounds_works_for_end_in_range() {
        check_buffer_bounds(200, 100, BufferSize::new(50));
        check_buffer_bounds(200, 100, BufferSize::new(100));
        check_buffer_bounds(u64::MAX, u64::MAX - 100, BufferSize::new(100));
        check_buffer_bounds(u64::MAX, 0, BufferSize::new(u64::MAX));
        check_buffer_bounds(u64::MAX, 1, BufferSize::new(u64::MAX - 1));
    }

    #[test]
    #[should_panic]
    fn check_buffer_bounds_panics_for_end_over_size() {
        check_buffer_bounds(200, 100, BufferSize::new(101));
    }

    #[test]
    #[should_panic]
    fn check_buffer_bounds_panics_for_end_wraparound() {
        check_buffer_bounds(u64::MAX, 1, BufferSize::new(u64::MAX));
    }
}

use crate::{
    util::align_to, Buffer, BufferAddress, BufferDescriptor, BufferSize, BufferSlice, BufferUsages,
    BufferViewMut, CommandEncoder, Device, MapMode,
};
use alloc::vec::Vec;
use core::fmt;
use std::sync::mpsc;

/// Efficiently performs many buffer writes by sharing and reusing temporary buffers.
///
/// Internally it uses a ring-buffer of staging buffers that are sub-allocated.
/// Its advantage over [`Queue::write_buffer_with()`] is that the individual allocations
/// are cheaper; `StagingBelt` is most useful when you are writing very many small pieces
/// of data. It can be understood as a sort of arena allocator.
///
/// Using a staging belt is slightly complicated, and generally goes as follows:
/// 1. Use [`StagingBelt::write_buffer()`] or [`StagingBelt::allocate()`] to allocate
///    buffer slices, then write your data to them.
/// 2. Call [`StagingBelt::finish()`].
/// 3. Submit all command encoders that were used in step 1.
/// 4. Call [`StagingBelt::recall()`].
///
/// [`Queue::write_buffer_with()`]: crate::Queue::write_buffer_with
pub struct StagingBelt {
    chunk_size: BufferAddress,
    /// Chunks into which we are accumulating data to be transferred.
    active_chunks: Vec<Chunk>,
    /// Chunks that have scheduled transfers already; they are unmapped and some
    /// command encoder has one or more commands with them as source.
    closed_chunks: Vec<Chunk>,
    /// Chunks that are back from the GPU and ready to be mapped for write and put
    /// into `active_chunks`.
    free_chunks: Vec<Chunk>,
    /// When closed chunks are mapped again, the map callback sends them here.
    sender: Exclusive<mpsc::Sender<Chunk>>,
    /// Free chunks are received here to be put on `self.free_chunks`.
    receiver: Exclusive<mpsc::Receiver<Chunk>>,
}

impl StagingBelt {
    /// Create a new staging belt.
    ///
    /// The `chunk_size` is the unit of internal buffer allocation; writes will be
    /// sub-allocated within each chunk. Therefore, for optimal use of memory, the
    /// chunk size should be:
    ///
    /// * larger than the largest single [`StagingBelt::write_buffer()`] operation;
    /// * 1-4 times less than the total amount of data uploaded per submission
    ///   (per [`StagingBelt::finish()`]); and
    /// * bigger is better, within these bounds.
    pub fn new(chunk_size: BufferAddress) -> Self {
        let (sender, receiver) = mpsc::channel();
        StagingBelt {
            chunk_size,
            active_chunks: Vec::new(),
            closed_chunks: Vec::new(),
            free_chunks: Vec::new(),
            sender: Exclusive::new(sender),
            receiver: Exclusive::new(receiver),
        }
    }

    /// Allocate a staging belt slice of `size` to be copied into the `target` buffer
    /// at the specified offset.
    ///
    /// The upload will be placed into the provided command encoder. This encoder
    /// must be submitted after [`StagingBelt::finish()`] is called and before
    /// [`StagingBelt::recall()`] is called.
    ///
    /// If the `size` is greater than the size of any free internal buffer, a new buffer
    /// will be allocated for it. Therefore, the `chunk_size` passed to [`StagingBelt::new()`]
    /// should ideally be larger than every such size.
    pub fn write_buffer(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        offset: BufferAddress,
        size: BufferSize,
        device: &Device,
    ) -> BufferViewMut<'_> {
        let slice_of_belt = self.allocate(
            size,
            const { BufferSize::new(crate::COPY_BUFFER_ALIGNMENT).unwrap() },
            device,
        );
        encoder.copy_buffer_to_buffer(
            slice_of_belt.buffer(),
            slice_of_belt.offset(),
            target,
            offset,
            size.get(),
        );
        slice_of_belt.get_mapped_range_mut()
    }

    /// Allocate a staging belt slice with the given `size` and `alignment` and return it.
    ///
    /// To use this slice, call [`BufferSlice::get_mapped_range_mut()`] and write your data into
    /// that [`BufferViewMut`].
    /// (The view must be dropped before [`StagingBelt::finish()`] is called.)
    ///
    /// You can then record your own GPU commands to perform with the slice,
    /// such as copying it to a texture or executing a compute shader that reads it (whereas
    /// [`StagingBelt::write_buffer()`] can only write to other buffers).
    /// All commands involving this slice must be submitted after
    /// [`StagingBelt::finish()`] is called and before [`StagingBelt::recall()`] is called.
    ///
    /// If the `size` is greater than the space available in any free internal buffer, a new buffer
    /// will be allocated for it. Therefore, the `chunk_size` passed to [`StagingBelt::new()`]
    /// should ideally be larger than every such size.
    ///
    /// The chosen slice will be positioned within the buffer at a multiple of `alignment`,
    /// which may be used to meet alignment requirements for the operation you wish to perform
    /// with the slice. This does not necessarily affect the alignment of the [`BufferViewMut`].
    pub fn allocate(
        &mut self,
        size: BufferSize,
        alignment: BufferSize,
        device: &Device,
    ) -> BufferSlice<'_> {
        assert!(
            alignment.get().is_power_of_two(),
            "alignment must be a power of two, not {alignment}"
        );
        // At minimum, we must have alignment sufficient to map the buffer.
        let alignment = alignment.get().max(crate::MAP_ALIGNMENT);

        let mut chunk = if let Some(index) = self
            .active_chunks
            .iter()
            .position(|chunk| chunk.can_allocate(size, alignment))
        {
            self.active_chunks.swap_remove(index)
        } else {
            self.receive_chunks(); // ensure self.free_chunks is up to date

            if let Some(index) = self
                .free_chunks
                .iter()
                .position(|chunk| chunk.can_allocate(size, alignment))
            {
                self.free_chunks.swap_remove(index)
            } else {
                Chunk {
                    buffer: device.create_buffer(&BufferDescriptor {
                        label: Some("(wgpu internal) StagingBelt staging buffer"),
                        size: self.chunk_size.max(size.get()),
                        usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    }),
                    offset: 0,
                }
            }
        };

        let allocation_offset = chunk.allocate(size, alignment);

        self.active_chunks.push(chunk);
        let chunk = self.active_chunks.last().unwrap();

        chunk
            .buffer
            .slice(allocation_offset..allocation_offset + size.get())
    }

    /// Prepare currently mapped buffers for use in a submission.
    ///
    /// This must be called before the command encoder(s) provided to
    /// [`StagingBelt::write_buffer()`] are submitted.
    ///
    /// At this point, all the partially used staging buffers are closed (cannot be used for
    /// further writes) until after [`StagingBelt::recall()`] is called *and* the GPU is done
    /// copying the data from them.
    pub fn finish(&mut self) {
        for chunk in self.active_chunks.drain(..) {
            chunk.buffer.unmap();
            self.closed_chunks.push(chunk);
        }
    }

    /// Recall all of the closed buffers back to be reused.
    ///
    /// This must only be called after the command encoder(s) provided to
    /// [`StagingBelt::write_buffer()`] are submitted. Additional calls are harmless.
    /// Not calling this as soon as possible may result in increased buffer memory usage.
    pub fn recall(&mut self) {
        self.receive_chunks();

        for chunk in self.closed_chunks.drain(..) {
            let sender = self.sender.get_mut().clone();
            chunk
                .buffer
                .clone()
                .slice(..)
                .map_async(MapMode::Write, move |_| {
                    let _ = sender.send(chunk);
                });
        }
    }

    /// Move all chunks that the GPU is done with (and are now mapped again)
    /// from `self.receiver` to `self.free_chunks`.
    fn receive_chunks(&mut self) {
        while let Ok(mut chunk) = self.receiver.get_mut().try_recv() {
            chunk.offset = 0;
            self.free_chunks.push(chunk);
        }
    }
}

impl fmt::Debug for StagingBelt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StagingBelt")
            .field("chunk_size", &self.chunk_size)
            .field("active_chunks", &self.active_chunks.len())
            .field("closed_chunks", &self.closed_chunks.len())
            .field("free_chunks", &self.free_chunks.len())
            .finish_non_exhaustive()
    }
}

struct Chunk {
    buffer: Buffer,
    offset: BufferAddress,
}

impl Chunk {
    fn can_allocate(&self, size: BufferSize, alignment: BufferAddress) -> bool {
        let alloc_start = align_to(self.offset, alignment);
        let alloc_end = alloc_start + size.get();

        alloc_end <= self.buffer.size()
    }

    fn allocate(&mut self, size: BufferSize, alignment: BufferAddress) -> BufferAddress {
        let alloc_start = align_to(self.offset, alignment);
        let alloc_end = alloc_start + size.get();

        assert!(alloc_end <= self.buffer.size());
        self.offset = alloc_end;
        alloc_start
    }
}

use exclusive::Exclusive;
mod exclusive {
    /// `Sync` wrapper that works by providing only exclusive access.
    ///
    /// See <https://doc.rust-lang.org/nightly/std/sync/struct.Exclusive.html>
    pub(super) struct Exclusive<T>(T);

    /// Safety: `&Exclusive` has no operations.
    unsafe impl<T> Sync for Exclusive<T> {}

    impl<T> Exclusive<T> {
        pub fn new(value: T) -> Self {
            Self(value)
        }

        pub fn get_mut(&mut self) -> &mut T {
            &mut self.0
        }
    }
}

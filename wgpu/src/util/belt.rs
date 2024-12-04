use crate::{
    util::align_to, Buffer, BufferAddress, BufferDescriptor, BufferSize, BufferUsages,
    BufferViewMut, CommandEncoder, Device, MapMode,
};
use std::fmt;
use std::sync::{mpsc, Arc};

struct Chunk {
    buffer: Arc<Buffer>,
    size: BufferAddress,
    offset: BufferAddress,
}

/// `Sync` wrapper that works by providing only exclusive access.
///
/// See https://doc.rust-lang.org/nightly/std/sync/struct.Exclusive.html
struct Exclusive<T>(T);

unsafe impl<T> Sync for Exclusive<T> {}

impl<T> Exclusive<T> {
    fn new(value: T) -> Self {
        Self(value)
    }

    fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

/// Efficiently performs many buffer writes by sharing and reusing temporary buffers.
///
/// Internally it uses a ring-buffer of staging buffers that are sub-allocated.
/// It has an advantage over [`Queue::write_buffer()`] in a way that it returns a mutable slice,
/// which you can fill to avoid an extra data copy.
///
/// Using a staging belt is slightly complicated, and generally goes as follows:
/// 1. Write to buffers that need writing to using [`StagingBelt::write_buffer()`].
/// 2. Call [`StagingBelt::finish()`].
/// 3. Submit all command encoders that were used in step 1.
/// 4. Call [`StagingBelt::recall()`].
///
/// [`Queue::write_buffer()`]: crate::Queue::write_buffer
pub struct StagingBelt {
    chunk_size: BufferAddress,
    /// Chunks into which we are accumulating data to be transferred.
    active_chunks: Vec<Chunk>,
    /// Chunks that have scheduled transfers already; they are unmapped and some
    /// command encoder has one or more `copy_buffer_to_buffer` commands with them
    /// as source.
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
        let (sender, receiver) = std::sync::mpsc::channel();
        StagingBelt {
            chunk_size,
            active_chunks: Vec::new(),
            closed_chunks: Vec::new(),
            free_chunks: Vec::new(),
            sender: Exclusive::new(sender),
            receiver: Exclusive::new(receiver),
        }
    }

    /// Allocate the staging belt slice of `size` to be uploaded into the `target` buffer
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
        let mut chunk = if let Some(index) = self
            .active_chunks
            .iter()
            .position(|chunk| chunk.offset + size.get() <= chunk.size)
        {
            self.active_chunks.swap_remove(index)
        } else {
            self.receive_chunks(); // ensure self.free_chunks is up to date

            if let Some(index) = self
                .free_chunks
                .iter()
                .position(|chunk| size.get() <= chunk.size)
            {
                self.free_chunks.swap_remove(index)
            } else {
                let size = self.chunk_size.max(size.get());
                Chunk {
                    buffer: Arc::new(device.create_buffer(&BufferDescriptor {
                        label: Some("(wgpu internal) StagingBelt staging buffer"),
                        size,
                        usage: BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
                        mapped_at_creation: true,
                    })),
                    size,
                    offset: 0,
                }
            }
        };

        encoder.copy_buffer_to_buffer(&chunk.buffer, chunk.offset, target, offset, size.get());
        let old_offset = chunk.offset;
        chunk.offset = align_to(chunk.offset + size.get(), crate::MAP_ALIGNMENT);

        self.active_chunks.push(chunk);
        self.active_chunks
            .last()
            .unwrap()
            .buffer
            .slice(old_offset..old_offset + size.get())
            .get_mapped_range_mut()
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

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

/// Staging belt is a machine that uploads data.
///
/// Internally it uses a ring-buffer of staging buffers that are sub-allocated.
/// It has an advantage over [`Queue::write_buffer`] in a way that it returns a mutable slice,
/// which you can fill to avoid an extra data copy.
///
/// Using a staging belt is slightly complicated, and generally goes as follows:
/// - Write to buffers that need writing to using [`StagingBelt::write_buffer`].
/// - Call `finish`.
/// - Submit all command encoders used with `StagingBelt::write_buffer`.
/// - Call `recall`
///
/// [`Queue::write_buffer`]: crate::Queue::write_buffer
pub struct StagingBelt {
    chunk_size: BufferAddress,
    /// Chunks that we are actively using for pending transfers at this moment.
    active_chunks: Vec<Chunk>,
    /// Chunks that have scheduled transfers already.
    closed_chunks: Vec<Chunk>,
    /// Chunks that are back from the GPU and ready to be used.
    free_chunks: Vec<Chunk>,
    sender: mpsc::Sender<Chunk>,
    receiver: mpsc::Receiver<Chunk>,
}

impl StagingBelt {
    /// Create a new staging belt.
    ///
    /// The `chunk_size` is the unit of internal buffer allocation.
    /// It's better when it's big, but ideally still 1-4 times less than
    /// the total amount of data uploaded per submission.
    pub fn new(chunk_size: BufferAddress) -> Self {
        let (sender, receiver) = mpsc::channel();
        StagingBelt {
            chunk_size,
            active_chunks: Vec::new(),
            closed_chunks: Vec::new(),
            free_chunks: Vec::new(),
            sender,
            receiver,
        }
    }

    /// Allocate the staging belt slice of `size` to be uploaded into the `target` buffer
    /// at the specified offset.
    ///
    /// The upload will be placed into the provided command encoder. This encoder
    /// must be submitted after `finish` is called and before `recall` is called.
    pub fn write_buffer(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &Buffer,
        offset: BufferAddress,
        size: BufferSize,
        device: &Device,
    ) -> BufferViewMut {
        let mut chunk = if let Some(index) = self
            .active_chunks
            .iter()
            .position(|chunk| chunk.offset + size.get() <= chunk.size)
        {
            self.active_chunks.swap_remove(index)
        } else if let Some(index) = self
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
    /// At this point, all the partially used staging buffers are closed until
    /// the GPU is done copying the data from them.
    pub fn finish(&mut self) {
        for chunk in self.active_chunks.drain(..) {
            chunk.buffer.unmap();
            self.closed_chunks.push(chunk);
        }
    }

    /// Recall all of the closed buffers back to be reused.
    ///
    /// This has to be called after the command encoders written to `write_buffer` are submitted!
    pub fn recall(&mut self) {
        while let Ok(mut chunk) = self.receiver.try_recv() {
            chunk.offset = 0;
            self.free_chunks.push(chunk);
        }

        let sender = &self.sender;
        for chunk in self.closed_chunks.drain(..) {
            let sender = sender.clone();
            chunk
                .buffer
                .clone()
                .slice(..)
                .map_async(MapMode::Write, move |_| {
                    let _ = sender.send(chunk);
                });
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

use crate::{
    Buffer, BufferAddress, BufferDescriptor, BufferSize, BufferUsage, BufferViewMut, CommandBuffer,
    CommandEncoder, CommandEncoderDescriptor, Device, MapMode,
};
use futures::{future::join_all, FutureExt};
use std::{borrow::Cow::Borrowed, future::Future, mem, sync::mpsc};

struct Chunk {
    buffer: Buffer,
    size: BufferAddress,
    offset: BufferAddress,
}

/// Staging belt is a machine that uploads data.
///
/// Internally it uses a ring-buffer of staging buffers that are sub-allocated.
/// It has an advantage over `Queue.write_buffer` in a way that it returns a mutable slice,
/// which you can fill to avoid an extra data copy.
pub struct StagingBelt {
    chunk_size: BufferAddress,
    encoder: CommandEncoder,
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
    pub fn new(chunk_size: BufferAddress, device: &Device) -> Self {
        let (sender, receiver) = mpsc::channel();
        StagingBelt {
            chunk_size,
            encoder: device.create_command_encoder(&CommandEncoderDescriptor::default()),
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
    /// The upload will only really be scheduled at the next `StagingBelt::flush` call.
    pub fn write_buffer(
        &mut self,
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
            #[cfg(not(target_arch = "wasm32"))]
            wgc::span!(_guard, INFO, "Creating chunk of size {}", size);
            Chunk {
                buffer: device.create_buffer(&BufferDescriptor {
                    label: Some(Borrowed("staging")),
                    size,
                    usage: BufferUsage::MAP_WRITE | BufferUsage::COPY_SRC,
                    mapped_at_creation: true,
                }),
                size,
                offset: 0,
            }
        };

        self.encoder
            .copy_buffer_to_buffer(&chunk.buffer, chunk.offset, target, offset, size.get());
        let old_offset = chunk.offset;
        chunk.offset += size.get();
        let remainder = chunk.offset % crate::COPY_BUFFER_ALIGNMENT;
        if remainder != 0 {
            chunk.offset += crate::COPY_BUFFER_ALIGNMENT - remainder;
        }

        self.active_chunks.push(chunk);
        self.active_chunks
            .last()
            .unwrap()
            .buffer
            .slice(old_offset..old_offset + size.get())
            .get_mapped_range_mut()
    }

    /// Produce a command buffer with all the accumulated transfers.
    ///
    /// At this point, all the partially used staging buffers are closed until
    /// the GPU is done copying the data from them.
    pub fn flush(&mut self, device: &Device) -> CommandBuffer {
        #[cfg(not(target_arch = "wasm32"))]
        wgc::span!(_guard, DEBUG, "Flushing chunks");

        for chunk in self.active_chunks.drain(..) {
            chunk.buffer.unmap();
            self.closed_chunks.push(chunk);
        }

        mem::replace(
            &mut self.encoder,
            device.create_command_encoder(&CommandEncoderDescriptor::default()),
        )
        .finish()
    }

    /// Recall all of the closed buffers back for re-usal.
    ///
    /// This has to be called after the command buffer produced by `flush` is submitted!
    pub fn recall(&mut self) -> impl Future<Output = ()> + Send {
        while let Ok(mut chunk) = self.receiver.try_recv() {
            chunk.offset = 0;
            self.free_chunks.push(chunk);
        }

        let sender_template = &self.sender;
        join_all(self.closed_chunks.drain(..).map(|chunk| {
            let sender = sender_template.clone();
            chunk
                .buffer
                .slice(..)
                .map_async(MapMode::Write)
                .inspect(move |_| sender.send(chunk).unwrap())
        }))
        .map(|_| ())
    }
}

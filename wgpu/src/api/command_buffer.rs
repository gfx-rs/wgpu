use crate::*;

/// Handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
///
/// Corresponds to [WebGPU `GPUCommandBuffer`](https://gpuweb.github.io/gpuweb/#command-buffer).
#[derive(Debug)]
pub struct CommandBuffer {
    pub(crate) buffer: dispatch::DispatchCommandBuffer,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(CommandBuffer: Send, Sync);

impl CommandBuffer {
    #[cfg(custom)]
    /// Returns custom implementation of CommandBuffer (if custom backend and is internally T)
    pub fn as_custom<T: custom::CommandBufferInterface>(&self) -> Option<&T> {
        self.buffer.as_custom()
    }
}

use std::sync::Arc;

use parking_lot::Mutex;

use crate::*;

/// Handle to a command buffer on the GPU.
///
/// A `CommandBuffer` represents a complete sequence of commands that may be submitted to a command
/// queue with [`Queue::submit`]. A `CommandBuffer` is obtained by recording a series of commands to
/// a [`CommandEncoder`] and then calling [`CommandEncoder::finish`].
///
/// Corresponds to [WebGPU `GPUCommandBuffer`](https://gpuweb.github.io/gpuweb/#command-buffer).
#[derive(Debug, Clone)]
pub struct CommandBuffer {
    pub(crate) inner: Arc<Mutex<Option<dispatch::DispatchCommandBuffer>>>,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(CommandBuffer: Send, Sync);

crate::cmp::impl_eq_ord_hash_arc_address!(CommandBuffer => .inner);

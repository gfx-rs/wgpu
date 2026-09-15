/*!
# Dos and Don'ts

A short list of common performance pitfalls and the patterns that avoid
them.

### Don't: create temporary mapped buffers when updating data

Instead, [`Queue::write_buffer`] and [`Queue::write_texture`] can be used
conveniently. If you are uploading a lot of data that is generated (as
opposed to already sitting in a data vector), it may be more efficient to
recycle staging buffers in a pool.

### Do: group resource bindings by change frequency, starting from the lowest

For example, put per-frame resources into bind group 0, per-pass resources
into bind group 1, and per-material resources into bind group 2. This allows
the WebGPU implementation to keep the other bindings intact, reducing state
changes.

### Don't: create many resources (buffers or textures) per frame

This puts pressure on the WebGPU memory allocator and tracker. Prefer
coalescing smaller resources into larger ones. For buffers, you can create a
large buffer and use different parts of it for different purposes. For
textures, consider texture atlases and arrays.

### Don't: submit many times per frame

There is a visible CPU cost per submission, and resources are tracked per
submission by the implementation. It is fine to have multiple
[`CommandBuffer`]s per submission, but the number of [`Queue::submit`] calls
should be limited to a few per frame (e.g. 1–5).
*/

use crate::{CommandBuffer, Queue};

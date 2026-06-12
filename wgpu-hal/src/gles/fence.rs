use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::Ordering;
use parking_lot::RwLock;

use glow::HasContext;

use crate::AtomicFenceValue;

#[derive(Debug)]
struct GLFence {
    // Since a fence can be `Copy`ed, there can exist some
    // cases where a fence could be destroyed while something
    // else is still using it. Therefore, while a function is
    // using this fence (and doesn't keep pending read locked),
    // it should clone the `Arc` to show it needs this to
    // stay alive.
    //
    // The arc should not be kept after a function has finished
    sync: Arc<glow::Fence>,
    value: crate::FenceValue,
}

#[derive(Debug)]
pub struct Fence {
    last_completed: AtomicFenceValue,
    pending: RwLock<Vec<GLFence>>,
    fence_behavior: wgt::GlFenceBehavior,
}

impl crate::DynFence for Fence {}

#[cfg(send_sync)]
unsafe impl Send for Fence {}
#[cfg(send_sync)]
unsafe impl Sync for Fence {}

impl Fence {
    pub fn new(options: &wgt::GlBackendOptions) -> Self {
        Self {
            last_completed: AtomicFenceValue::new(0),
            pending: RwLock::new(Vec::new()),
            fence_behavior: options.fence_behavior,
        }
    }

    pub fn signal(
        &self,
        gl: &glow::Context,
        value: crate::FenceValue,
    ) -> Result<(), crate::DeviceError> {
        if self.fence_behavior.is_auto_finish() {
            self.last_completed.store(value, Ordering::Release);
            return Ok(());
        }

        let sync = unsafe { gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) }
            .map_err(|_| crate::DeviceError::OutOfMemory)?;
        self.pending.write().push(GLFence {
            sync: Arc::new(sync),
            value,
        });

        Ok(())
    }

    pub fn satisfied(&self, value: crate::FenceValue) -> bool {
        self.last_completed.load(Ordering::Acquire) >= value
    }

    pub fn get_latest(&self, gl: &glow::Context) -> crate::FenceValue {
        let mut max_value = self.last_completed.load(Ordering::Acquire);

        if self.fence_behavior.is_auto_finish() {
            return max_value;
        }

        let pending = self.pending.read();

        for gl_fence in pending.iter() {
            if gl_fence.value <= max_value {
                // We already know this was good, no need to check again
                continue;
            }
            // We have pending `read` locked, so we shouldn't have to clone it.
            let status = unsafe { gl.get_sync_status(*gl_fence.sync) };
            if status == glow::SIGNALED {
                max_value = gl_fence.value;
            } else {
                // Anything after the first unsignalled is guaranteed to also be unsignalled
                break;
            }
        }

        // Track the latest value, to save ourselves some querying later
        self.last_completed.fetch_max(max_value, Ordering::AcqRel);

        max_value
    }

    pub fn maintain(&self, gl: &glow::Context) {
        if self.fence_behavior.is_auto_finish() {
            return;
        }

        let latest = self.get_latest(gl);
        let mut pending = self.pending.write();
        pending.retain_mut(|gl_fence| {
            if gl_fence.value > latest {
                true
            } else if let Some(fence) = Arc::get_mut(&mut gl_fence.sync) {
                unsafe {
                    gl.delete_sync(*fence);
                }
                false
            } else {
                // Another function is currently using this value. In general, these should finish
                // very quickly (for wait because the fence should already be signaled, an all
                // others are just fast), but submit should be very fast, so we shouldn't block on
                // this.
                true
            }
        });
    }

    pub fn wait(
        &self,
        gl: &glow::Context,
        wait_value: crate::FenceValue,
        timeout_ns: u32,
    ) -> Result<bool, crate::DeviceError> {
        let last_completed = self.last_completed.load(Ordering::Acquire);

        if self.fence_behavior.is_auto_finish() {
            return Ok(last_completed >= wait_value);
        }

        // We already know this fence has been signalled to that value. Return signalled.
        if last_completed >= wait_value {
            return Ok(true);
        }

        let pending = self.pending.read();

        // Find a matching fence
        let gl_fence = pending.iter().find(|gl_fence| gl_fence.value >= wait_value);

        let Some(gl_fence) = gl_fence else {
            log::warn!("Tried to wait for {wait_value} but that value has not been signalled yet");
            return Ok(false);
        };

        // clone to show we're using the fence
        let sync = gl_fence.sync.clone();
        let fence_value = gl_fence.value;

        drop(pending);

        let status = unsafe {
            gl.client_wait_sync(
                *sync,
                glow::SYNC_FLUSH_COMMANDS_BIT,
                timeout_ns.min(i32::MAX as u32) as i32,
            )
        };

        drop(sync);

        let signalled = match status {
            glow::ALREADY_SIGNALED | glow::CONDITION_SATISFIED => true,
            glow::TIMEOUT_EXPIRED | glow::WAIT_FAILED => false,
            _ => {
                log::warn!("Unexpected result from client_wait_sync: {status}");
                false
            }
        };

        if signalled {
            self.last_completed.fetch_max(fence_value, Ordering::AcqRel);
        }

        Ok(signalled)
    }

    pub fn destroy(self, gl: &glow::Context) {
        if self.fence_behavior.is_auto_finish() {
            return;
        }

        for gl_fence in self.pending.into_inner() {
            unsafe {
                gl.delete_sync(
                    Arc::into_inner(gl_fence.sync)
                        .expect("A function has failed to drop all its references to this"),
                );
            }
        }
    }
}

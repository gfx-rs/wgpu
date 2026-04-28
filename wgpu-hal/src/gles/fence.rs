use alloc::{sync::Arc, vec::Vec};
use core::sync::atomic::Ordering;
use parking_lot::RwLock;

use glow::HasContext;

use crate::AtomicFenceValue;

#[derive(Debug)]
struct GLFence {
    // Since a fence can be `Copy`ed, there can exist some
    // cases where (without proper synchronisation),
    // a fence could be destroyed while something else is
    // still using it. Therefore, while a function is
    // using this fence, it should read this. (write
    // should be done when destroying it)
    //
    // The arc should not be kept after a function has finished
    sync: Arc<RwLock<glow::Fence>>,
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
            sync: Arc::new(RwLock::new(sync)),
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
            let fence = gl_fence.sync.read();
            if gl_fence.value <= max_value {
                // We already know this was good, no need to check again
                continue;
            }
            let status = unsafe { gl.get_sync_status(*fence) };
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
        for gl_fence in pending.iter() {
            // We don't need to keep around this lock until after the retain - we need to make
            // sure nothing is using it by writing to it, but any new references must come
            // from `self.pending`, which is write-locked, so nothing else can take a
            // copy of this value
            let sync = *gl_fence.sync.write();

            if gl_fence.value <= latest {
                unsafe {
                    gl.delete_sync(sync);
                }
            }
        }
        pending.retain(|gl_fence| gl_fence.value > latest);
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
        let gl_fence = pending
            .iter()
            // Greater or equal as an abundance of caution, but there should be one fence per value
            .find(|gl_fence| gl_fence.value >= wait_value);

        let Some(gl_fence) = gl_fence else {
            log::warn!("Tried to wait for {wait_value} but that value has not been signalled yet");
            return Ok(false);
        };

        // We should have found a fence with the exact value.
        debug_assert_eq!(gl_fence.value, wait_value);

        let sync = gl_fence.sync.clone();

        drop(pending);

        let status = unsafe {
            gl.client_wait_sync(
                *sync.read(),
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
            self.last_completed.fetch_max(wait_value, Ordering::AcqRel);
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
                        .expect("A function has failed to drop all its references to this")
                        .into_inner(),
                );
            }
        }
    }
}

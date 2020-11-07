/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::CommandBuffer;
use crate::{
    hub::GfxBackend, id::DeviceId, track::TrackerSet, FastHashMap, PrivateFeatures, Stored,
    SubmissionIndex,
};

use hal::{command::CommandBuffer as _, device::Device as _, pool::CommandPool as _};
use parking_lot::Mutex;

use std::thread;

const GROW_AMOUNT: usize = 20;

#[derive(Debug)]
struct CommandPool<B: hal::Backend> {
    raw: B::CommandPool,
    total: usize,
    available: Vec<B::CommandBuffer>,
    pending: Vec<(B::CommandBuffer, SubmissionIndex)>,
}

impl<B: hal::Backend> CommandPool<B> {
    fn maintain(&mut self, last_done_index: SubmissionIndex) {
        for i in (0..self.pending.len()).rev() {
            if self.pending[i].1 <= last_done_index {
                let (cmd_buf, index) = self.pending.swap_remove(i);
                log::trace!(
                    "recycling comb submitted in {} when {} is last done",
                    index,
                    last_done_index,
                );
                self.recycle(cmd_buf);
            }
        }
    }

    fn recycle(&mut self, mut raw: B::CommandBuffer) {
        unsafe {
            raw.reset(false);
        }
        self.available.push(raw);
    }

    fn allocate(&mut self) -> B::CommandBuffer {
        if self.available.is_empty() {
            self.total += GROW_AMOUNT;
            unsafe {
                self.raw.allocate(
                    GROW_AMOUNT,
                    hal::command::Level::Primary,
                    &mut self.available,
                )
            };
        }
        self.available.pop().unwrap()
    }
}

#[derive(Debug)]
struct Inner<B: hal::Backend> {
    pools: FastHashMap<thread::ThreadId, CommandPool<B>>,
}

#[derive(Debug)]
pub struct CommandAllocator<B: hal::Backend> {
    queue_family: hal::queue::QueueFamilyId,
    internal_thread_id: thread::ThreadId,
    inner: Mutex<Inner<B>>,
}

impl<B: GfxBackend> CommandAllocator<B> {
    pub(crate) fn allocate(
        &self,
        device_id: Stored<DeviceId>,
        device: &B::Device,
        limits: wgt::Limits,
        private_features: PrivateFeatures,
        #[cfg(feature = "trace")] enable_tracing: bool,
    ) -> CommandBuffer<B> {
        //debug_assert_eq!(device_id.backend(), B::VARIANT);
        let thread_id = thread::current().id();
        let mut inner = self.inner.lock();

        let init = inner
            .pools
            .entry(thread_id)
            .or_insert_with(|| CommandPool {
                raw: unsafe {
                    log::info!("Starting on thread {:?}", thread_id);
                    device.create_command_pool(
                        self.queue_family,
                        hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
                    )
                }
                .unwrap(),
                total: 0,
                available: Vec::new(),
                pending: Vec::new(),
            })
            .allocate();

        CommandBuffer {
            raw: vec![init],
            is_recording: true,
            recorded_thread_id: thread_id,
            device_id,
            trackers: TrackerSet::new(B::VARIANT),
            used_swap_chain: None,
            limits,
            private_features,
            #[cfg(feature = "trace")]
            commands: if enable_tracing {
                Some(Vec::new())
            } else {
                None
            },
        }
    }
}

impl<B: hal::Backend> CommandAllocator<B> {
    pub fn new(queue_family: hal::queue::QueueFamilyId, device: &B::Device) -> Self {
        let internal_thread_id = thread::current().id();
        log::info!("Starting on (internal) thread {:?}", internal_thread_id);
        let mut pools = FastHashMap::default();
        pools.insert(
            internal_thread_id,
            CommandPool {
                raw: unsafe {
                    device
                        .create_command_pool(
                            queue_family,
                            hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
                        )
                        .unwrap()
                },
                total: 0,
                available: Vec::new(),
                pending: Vec::new(),
            },
        );
        CommandAllocator {
            queue_family,
            internal_thread_id,
            inner: Mutex::new(Inner { pools }),
        }
    }

    fn allocate_for_thread_id(&self, thread_id: thread::ThreadId) -> B::CommandBuffer {
        let mut inner = self.inner.lock();
        inner.pools.get_mut(&thread_id).unwrap().allocate()
    }

    pub fn allocate_internal(&self) -> B::CommandBuffer {
        self.allocate_for_thread_id(self.internal_thread_id)
    }

    pub fn extend(&self, cmd_buf: &CommandBuffer<B>) -> B::CommandBuffer {
        self.allocate_for_thread_id(cmd_buf.recorded_thread_id)
    }

    pub fn discard_internal(&self, raw: B::CommandBuffer) {
        let mut inner = self.inner.lock();
        inner
            .pools
            .get_mut(&self.internal_thread_id)
            .unwrap()
            .recycle(raw);
    }

    pub fn discard(&self, mut cmd_buf: CommandBuffer<B>) {
        cmd_buf.trackers.clear();
        let mut inner = self.inner.lock();
        let pool = inner.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();
        for raw in cmd_buf.raw {
            pool.recycle(raw);
        }
    }

    pub fn after_submit_internal(&self, raw: B::CommandBuffer, submit_index: SubmissionIndex) {
        let mut inner = self.inner.lock();
        inner
            .pools
            .get_mut(&self.internal_thread_id)
            .unwrap()
            .pending
            .push((raw, submit_index));
    }

    pub fn after_submit(&self, cmd_buf: CommandBuffer<B>, submit_index: SubmissionIndex) {
        // Record this command buffer as pending
        let mut inner = self.inner.lock();
        inner
            .pools
            .get_mut(&cmd_buf.recorded_thread_id)
            .unwrap()
            .pending
            .extend(cmd_buf.raw.into_iter().map(|raw| (raw, submit_index)));
    }

    pub fn maintain(&self, device: &B::Device, last_done_index: SubmissionIndex) {
        let mut inner = self.inner.lock();
        let mut remove_threads = Vec::new();
        for (&thread_id, pool) in inner.pools.iter_mut() {
            pool.maintain(last_done_index);
            if pool.total == pool.available.len() && thread_id != self.internal_thread_id {
                assert!(pool.pending.is_empty());
                remove_threads.push(thread_id);
            }
        }
        for thread_id in remove_threads {
            log::info!("Removing from thread {:?}", thread_id);
            let mut pool = inner.pools.remove(&thread_id).unwrap();
            unsafe {
                pool.raw.free(pool.available);
                device.destroy_command_pool(pool.raw);
            }
        }
    }

    pub fn destroy(self, device: &B::Device) {
        let mut inner = self.inner.lock();
        for (_, mut pool) in inner.pools.drain() {
            while let Some((raw, _)) = pool.pending.pop() {
                pool.recycle(raw);
            }
            if pool.total != pool.available.len() {
                log::error!(
                    "Some command buffers are still recorded, only tracking {} / {}",
                    pool.available.len(),
                    pool.total
                );
            }
            unsafe {
                pool.raw.free(pool.available);
                device.destroy_command_pool(pool.raw);
            }
        }
    }
}

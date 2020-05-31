/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::CommandBuffer;
use crate::{
    hub::GfxBackend, id::DeviceId, track::TrackerSet, LifeGuard, PrivateFeatures, Stored,
    SubmissionIndex,
};

use hal::{command::CommandBuffer as _, device::Device as _, pool::CommandPool as _};
use parking_lot::Mutex;

use std::{collections::HashMap, sync::atomic::Ordering, thread};

const GROW_AMOUNT: usize = 20;

#[derive(Debug)]
struct CommandPool<B: hal::Backend> {
    raw: B::CommandPool,
    total: usize,
    available: Vec<B::CommandBuffer>,
    pending: Vec<CommandBuffer<B>>,
}

impl<B: hal::Backend> CommandPool<B> {
    fn maintain(&mut self, lowest_active_index: SubmissionIndex) {
        for i in (0..self.pending.len()).rev() {
            let index = self.pending[i]
                .life_guard
                .submission_index
                .load(Ordering::Acquire);
            if index < lowest_active_index {
                let cmd_buf = self.pending.swap_remove(i);
                log::trace!(
                    "recycling comb submitted in {} when {} is lowest active",
                    index,
                    lowest_active_index,
                );
                self.recycle(cmd_buf);
            }
        }
    }

    fn recycle(&mut self, cmd_buf: CommandBuffer<B>) {
        for mut raw in cmd_buf.raw {
            unsafe {
                raw.reset(false);
            }
            self.available.push(raw);
        }
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
    pools: HashMap<thread::ThreadId, CommandPool<B>>,
}

#[derive(Debug)]
pub struct CommandAllocator<B: hal::Backend> {
    queue_family: hal::queue::QueueFamilyId,
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
            life_guard: LifeGuard::new(),
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
    pub fn new(queue_family: hal::queue::QueueFamilyId) -> Self {
        CommandAllocator {
            queue_family,
            inner: Mutex::new(Inner {
                pools: HashMap::new(),
            }),
        }
    }

    pub fn extend(&self, cmd_buf: &CommandBuffer<B>) -> B::CommandBuffer {
        let mut inner = self.inner.lock();
        inner
            .pools
            .get_mut(&cmd_buf.recorded_thread_id)
            .unwrap()
            .allocate()
    }

    pub fn discard(&self, mut cmd_buf: CommandBuffer<B>) {
        cmd_buf.trackers.clear();
        let mut inner = self.inner.lock();
        inner
            .pools
            .get_mut(&cmd_buf.recorded_thread_id)
            .unwrap()
            .recycle(cmd_buf);
    }

    pub fn after_submit(&self, mut cmd_buf: CommandBuffer<B>, submit_index: SubmissionIndex) {
        cmd_buf.trackers.clear();
        cmd_buf
            .life_guard
            .submission_index
            .store(submit_index, Ordering::Release);

        // Record this command buffer as pending
        let mut inner = self.inner.lock();
        inner
            .pools
            .get_mut(&cmd_buf.recorded_thread_id)
            .unwrap()
            .pending
            .push(cmd_buf);
    }

    pub fn maintain(&self, device: &B::Device, lowest_active_index: SubmissionIndex) {
        let mut inner = self.inner.lock();
        let mut remove_threads = Vec::new();
        for (thread_id, pool) in inner.pools.iter_mut() {
            pool.maintain(lowest_active_index);
            if pool.total == pool.available.len() {
                assert!(pool.pending.is_empty());
                remove_threads.push(*thread_id);
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
            while let Some(cmd_buf) = pool.pending.pop() {
                pool.recycle(cmd_buf);
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

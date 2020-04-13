/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use super::CommandBuffer;
use crate::{
    hub::GfxBackend,
    id::DeviceId,
    track::TrackerSet,
    Features,
    LifeGuard,
    Stored,
    SubmissionIndex,
};

use hal::{command::CommandBuffer as _, device::Device as _, pool::CommandPool as _};
use parking_lot::Mutex;

use std::{collections::HashMap, sync::atomic::Ordering, thread};

#[derive(Debug)]
struct CommandPool<B: hal::Backend> {
    raw: B::CommandPool,
    available: Vec<B::CommandBuffer>,
    pending: Vec<CommandBuffer<B>>,
}

impl<B: hal::Backend> CommandPool<B> {
    fn maintain(&mut self, lowest_active_index: SubmissionIndex) {
        for i in (0 .. self.pending.len()).rev() {
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
            unsafe { self.raw.allocate(20, hal::command::Level::Primary, &mut self.available) };
        }
        self.available.pop().unwrap()
    }
}

#[derive(Debug)]
struct Inner<B: hal::Backend> {
    // TODO: Currently pools from threads that are stopped or no longer call into wgpu will never be
    // cleaned up.
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
        features: Features,
        lowest_active_index: SubmissionIndex,
    ) -> CommandBuffer<B> {
        //debug_assert_eq!(device_id.backend(), B::VARIANT);
        let thread_id = thread::current().id();
        let mut inner = self.inner.lock();

        let pool = inner.pools.entry(thread_id).or_insert_with(|| CommandPool {
            raw: unsafe {
                device.create_command_pool(
                    self.queue_family,
                    hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
                )
            }
            .unwrap(),
            available: Vec::new(),
            pending: Vec::new(),
        });

        // Recycle completed command buffers
        pool.maintain(lowest_active_index);

        let init = pool.allocate();

        CommandBuffer {
            raw: vec![init],
            is_recording: true,
            recorded_thread_id: thread_id,
            device_id,
            life_guard: LifeGuard::new(),
            trackers: TrackerSet::new(B::VARIANT),
            used_swap_chain: None,
            features,
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
        let pool = inner.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();

        if pool.available.is_empty() {
            unsafe { pool.raw.allocate(20, hal::command::Level::Primary, &mut pool.available) };
        }

        pool.available.pop().unwrap()
    }

    pub fn discard(&self, mut cmd_buf: CommandBuffer<B>) {
        cmd_buf.trackers.clear();
        self.inner
            .lock()
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
        let pool = inner.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();
        pool.pending.push(cmd_buf);
    }

    pub fn destroy(self, device: &B::Device) {
        let mut inner = self.inner.lock();
        for (_, mut pool) in inner.pools.drain() {
            while let Some(cmd_buf) = pool.pending.pop() {
                pool.recycle(cmd_buf);
            }
            unsafe {
                pool.raw.free(pool.available);
                device.destroy_command_pool(pool.raw);
            }
        }
    }
}

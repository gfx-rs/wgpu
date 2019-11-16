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
}

impl<B: hal::Backend> CommandPool<B> {
    fn allocate(&mut self) -> B::CommandBuffer {
        if self.available.is_empty() {
            let extra = unsafe { self.raw.allocate_vec(20, hal::command::Level::Primary) };
            self.available.extend(extra);
        }

        self.available.pop().unwrap()
    }
}

#[derive(Debug)]
struct Inner<B: hal::Backend> {
    pools: HashMap<thread::ThreadId, CommandPool<B>>,
    pending: Vec<CommandBuffer<B>>,
}

impl<B: hal::Backend> Inner<B> {
    fn recycle(&mut self, cmd_buf: CommandBuffer<B>) {
        let pool = self.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();
        for mut raw in cmd_buf.raw {
            unsafe {
                raw.reset(false);
            }
            pool.available.push(raw);
        }
    }
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
        });
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
                pending: Vec::new(),
            }),
        }
    }

    pub fn extend(&self, cmd_buf: &CommandBuffer<B>) -> B::CommandBuffer {
        let mut inner = self.inner.lock();
        let pool = inner.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();

        if pool.available.is_empty() {
            let extra = unsafe { pool.raw.allocate_vec(20, hal::command::Level::Primary) };
            pool.available.extend(extra);
        }

        pool.available.pop().unwrap()
    }

    pub fn after_submit(&self, mut cmd_buf: CommandBuffer<B>, submit_index: SubmissionIndex) {
        cmd_buf.trackers.clear();
        cmd_buf
            .life_guard
            .submission_index
            .store(submit_index, Ordering::Release);
        self.inner.lock().pending.push(cmd_buf);
    }

    pub fn maintain(&self, last_done: SubmissionIndex) {
        let mut inner = self.inner.lock();
        for i in (0 .. inner.pending.len()).rev() {
            let index = inner.pending[i]
                .life_guard
                .submission_index
                .load(Ordering::Acquire);
            if index <= last_done {
                let cmd_buf = inner.pending.swap_remove(i);
                log::trace!(
                    "recycling comb submitted in {} when {} is done",
                    index,
                    last_done
                );
                inner.recycle(cmd_buf);
            }
        }
    }

    pub fn destroy(self, device: &B::Device) {
        let mut inner = self.inner.lock();
        while let Some(cmd_buf) = inner.pending.pop() {
            inner.recycle(cmd_buf);
        }
        for (_, mut pool) in inner.pools.drain() {
            unsafe {
                pool.raw.free(pool.available);
                device.destroy_command_pool(pool.raw);
            }
        }
    }
}

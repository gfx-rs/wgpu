use super::CommandBuffer;
use {DeviceId, Stored};

use hal::command::RawCommandBuffer;
use hal::pool::RawCommandPool;
use hal::{self, Device};

use std::collections::HashMap;
//TODO: use `parking_lot::Mutex`?
use std::sync::Mutex;
use std::thread;

struct CommandPool<B: hal::Backend> {
    raw: B::CommandPool,
    available: Vec<B::CommandBuffer>,
}

impl<B: hal::Backend> CommandPool<B> {
    fn allocate(&mut self) -> B::CommandBuffer {
        if self.available.is_empty() {
            let extra = self.raw.allocate(20, hal::command::RawLevel::Primary);
            self.available.extend(extra);
        }

        self.available.pop().unwrap()
    }
}

struct Inner<B: hal::Backend> {
    pools: HashMap<thread::ThreadId, CommandPool<B>>,
    fences: Vec<B::Fence>,
    pending: Vec<CommandBuffer<B>>,
}

impl<B: hal::Backend> Inner<B> {
    fn recycle(&mut self, cmd_buf: CommandBuffer<B>) {
        let pool = self.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();
        for mut raw in cmd_buf.raw {
            raw.reset(false);
            pool.available.push(raw);
        }
        self.fences.push(cmd_buf.fence);
    }
}

pub struct CommandAllocator<B: hal::Backend> {
    queue_family: hal::queue::QueueFamilyId,
    inner: Mutex<Inner<B>>,
}

impl<B: hal::Backend> CommandAllocator<B> {
    pub fn new(queue_family: hal::queue::QueueFamilyId) -> Self {
        CommandAllocator {
            queue_family,
            inner: Mutex::new(Inner {
                pools: HashMap::new(),
                fences: Vec::new(),
                pending: Vec::new(),
            }),
        }
    }

    pub fn allocate(
        &self, device_id: DeviceId, device: &B::Device
    ) -> CommandBuffer<B> {
        let thread_id = thread::current().id();
        let mut inner = self.inner.lock().unwrap();

        let fence = match inner.fences.pop() {
            Some(fence) => {
                device.reset_fence(&fence);
                fence
            }
            None => {
                device.create_fence(false)
            }
        };

        let pool = inner.pools.entry(thread_id).or_insert_with(|| CommandPool {
            raw: device.create_command_pool(
                self.queue_family,
                hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
            ),
            available: Vec::new(),
        });
        let init = pool.allocate();

        CommandBuffer {
            raw: vec![init],
            fence,
            recorded_thread_id: thread_id,
            device_id: Stored(device_id),
        }
    }

    pub fn extend(&self, cmd_buf: &CommandBuffer<B>) -> B::CommandBuffer {
        let mut inner = self.inner.lock().unwrap();
        let pool = inner.pools.get_mut(&cmd_buf.recorded_thread_id).unwrap();

        if pool.available.is_empty() {
            let extra = pool.raw.allocate(20, hal::command::RawLevel::Primary);
            pool.available.extend(extra);
        }

        pool.available.pop().unwrap()
    }

    pub fn submit(&self, cmd_buf: CommandBuffer<B>) {
        self.inner.lock().unwrap().pending.push(cmd_buf);
    }

    pub fn recycle(&self, cmd_buf: CommandBuffer<B>) {
        self.inner.lock().unwrap().recycle(cmd_buf);
    }

    pub fn maintain(&self, device: &B::Device) {
        let mut inner = self.inner.lock().unwrap();
        for i in (0..inner.pending.len()).rev() {
            if device.get_fence_status(&inner.pending[i].fence) {
                let cmd_buf = inner.pending.swap_remove(i);
                inner.recycle(cmd_buf);
            }
        }
    }
}

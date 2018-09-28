use super::CommandBuffer;

use hal::command::RawCommandBuffer;
use hal::pool::RawCommandPool;
use hal::{self, Device};

use std::collections::HashMap;
//TODO: use `parking_lot::Mutex`?
use std::sync::Mutex;
use std::thread;

struct CommandPool<B: hal::Backend> {
    raw: B::CommandPool,
    available: Vec<CommandBuffer<B>>,
}

pub struct Inner<B: hal::Backend> {
    pools: HashMap<thread::ThreadId, CommandPool<B>>,
    pending: Vec<CommandBuffer<B>>,
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
                pending: Vec::new(),
            }),
        }
    }

    pub fn allocate(&self, device: &B::Device) -> CommandBuffer<B> {
        let thread_id = thread::current().id();
        let mut inner = self.inner.lock().unwrap();
        let pool = inner.pools.entry(thread_id).or_insert_with(|| CommandPool {
            raw: device.create_command_pool(
                self.queue_family,
                hal::pool::CommandPoolCreateFlags::RESET_INDIVIDUAL,
            ),
            available: Vec::new(),
        });

        if let Some(cmd_buf) = pool.available.pop() {
            device.reset_fence(&cmd_buf.fence);
            return cmd_buf;
        }

        for raw in pool.raw.allocate(20, hal::command::RawLevel::Primary) {
            pool.available.push(CommandBuffer {
                raw,
                fence: device.create_fence(false),
                recorded_thread_id: thread_id,
            });
        }
        pool.available.pop().unwrap()
    }

    pub fn submit(&self, cmd_buf: CommandBuffer<B>) {
        self.inner.lock().unwrap().pending.push(cmd_buf);
    }

    pub fn recycle(&self, mut cmd_buf: CommandBuffer<B>) {
        cmd_buf.raw.reset(false);
        self.inner
            .lock()
            .unwrap()
            .pools
            .get_mut(&cmd_buf.recorded_thread_id)
            .unwrap()
            .available
            .push(cmd_buf);
    }

    pub fn maintain(&self, device: &B::Device) {
        let mut inner = self.inner.lock().unwrap();
        for i in (0..inner.pending.len()).rev() {
            if device.get_fence_status(&inner.pending[i].fence) {
                let cmd_buf = inner.pending.swap_remove(i);
                inner
                    .pools
                    .get_mut(&cmd_buf.recorded_thread_id)
                    .unwrap()
                    .available
                    .push(cmd_buf);
            }
        }
    }
}

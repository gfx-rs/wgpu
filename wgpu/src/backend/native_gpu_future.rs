use crate::BufferAddress;
use parking_lot::Mutex;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

enum WakerOrResult<T> {
    Waker(Waker),
    Result(T),
}

/// A Future that can poll the wgpu::Device
pub struct GpuFuture<T> {
    data: Arc<Data<T>>,
}

pub enum OpaqueData {}

struct Data<T> {
    buffer_id: wgc::id::BufferId,
    size: BufferAddress,
    waker_or_result: Mutex<Option<WakerOrResult<T>>>,
}

/// A completion handle to set the result on a GpuFuture
pub struct GpuFutureCompletion<T> {
    data: Arc<Data<T>>,
}

impl<T> Future for GpuFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        let mut waker_or_result = self.into_ref().get_ref().data.waker_or_result.lock();

        match waker_or_result.take() {
            Some(WakerOrResult::Result(res)) => Poll::Ready(res),
            _ => {
                *waker_or_result = Some(WakerOrResult::Waker(context.waker().clone()));
                Poll::Pending
            }
        }
    }
}

impl<T> GpuFutureCompletion<T> {
    pub fn complete(self, value: T) {
        let mut waker_or_result = self.data.waker_or_result.lock();

        match waker_or_result.replace(WakerOrResult::Result(value)) {
            Some(WakerOrResult::Waker(waker)) => waker.wake(),
            None => {}
            Some(WakerOrResult::Result(_)) => {
                // Drop before panicking. Not sure if this is necessary, but it makes me feel better.
                drop(waker_or_result);
                unreachable!()
            }
        };
    }

    pub(crate) fn to_raw(self) -> *mut OpaqueData {
        Arc::into_raw(self.data) as _
    }

    pub(crate) unsafe fn from_raw(this: *mut OpaqueData) -> Self {
        Self {
            data: Arc::from_raw(this as _),
        }
    }

    pub(crate) fn get_buffer_info(&self) -> (wgc::id::BufferId, BufferAddress) {
        (self.data.buffer_id, self.data.size)
    }
}

pub(crate) fn new_gpu_future<T>(
    buffer_id: wgc::id::BufferId,
    size: BufferAddress,
) -> (GpuFuture<T>, GpuFutureCompletion<T>) {
    let data = Arc::new(Data {
        buffer_id,
        size,
        waker_or_result: Mutex::new(None),
    });

    (
        GpuFuture {
            data: Arc::clone(&data),
        },
        GpuFutureCompletion { data },
    )
}

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

struct GpuFutureInner<T> {
    id: wgc::id::DeviceId,
    result: Option<T>,
    waker: Option<Waker>,
}

/// A Future that can poll the wgpu::Device
pub struct GpuFuture<T> {
    inner: Arc<Mutex<GpuFutureInner<T>>>,
}

/// A completion handle to set the result on a GpuFuture
pub struct GpuFutureCompletion<T> {
    inner: Arc<Mutex<GpuFutureInner<T>>>,
}

impl<T> Future for GpuFuture<T>
{
    type Output = T;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        // grab a clone of the Arc
        let arc = Arc::clone(&self.get_mut().inner);

        // grab the device id and set the waker, but release the lock, so that the native callback can write to it
        let device_id = {
            let mut inner = arc.lock().unwrap();
            inner.waker.replace(context.waker().clone());
            inner.id
        };

        // polling the device should trigger the callback
        wgn::wgpu_device_poll(device_id, true);

        // now take the lock again, and check whether the future is complete
        let mut inner = arc.lock().unwrap();
        match inner.result.take() {
            Some(value) => Poll::Ready(value),
            _ => Poll::Pending,
        }
    }
}

impl<T> GpuFutureCompletion<T> {
    pub fn complete(self, value: T) {
        let mut inner = self.inner.lock().unwrap();
        inner.result.replace(value);
        if let Some(waker) = &inner.waker {
            waker.wake_by_ref();
        }
    }
}

pub(crate) fn new_gpu_future<T>(id: wgc::id::DeviceId) -> (GpuFuture<T>, GpuFutureCompletion<T>) {
    let inner = Arc::new(Mutex::new(GpuFutureInner {
        id,
        result: None,
        waker: None,
    }));

    (
        GpuFuture {
            inner: inner.clone(),
        },
        GpuFutureCompletion { inner },
    )
}

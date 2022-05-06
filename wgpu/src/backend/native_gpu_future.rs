//! Futures that can be resolved when the GPU completes a task.
//!
//! This module defines the [`GpuFuture`] and [`GpuFutureCompletion`]
//! types, which `wgpu` uses to communicate to users when GPU
//! operations have completed, and when resources are ready to access.
//! This is only used by the `direct` back end, not on the web.
//!
//! The life cycle of a `GpuFuture` is as follows:
//!
//! -   Calling [`new_gpu_future`] constructs a paired `GpuFuture` and
//!     `GpuFutureCompletion`.
//!
//! -   Calling [`complete(v)`] on a `GpuFutureCompletion` marks its
//!     paired `GpuFuture` as ready with value `v`. This also wakes
//!     the most recent [`Waker`] the future was polled with, if any.
//!
//! -   Polling a `GpuFuture` either returns `v` if it is ready, or
//!     saves the `Waker` passed to [`Future::poll`], to be awoken
//!     when `complete` is called on the paired `GpuFutureCompletion`.
//!
//! ## Communicating with `wgpu_core`
//!
//! The `wgpu_core` crate uses various specialized callback types,
//! like [`wgpu_core::resource::BufferMapOperation`] for reporting
//! buffers that are ready to map, or
//! [`wgpu_core::device::queue::SubmittedWorkDoneClosure`] for
//! reporting the completion of submitted commands. To support FFI
//! bindings, these are unsafe, low-level structures that usually have
//! a function pointer and a untyped, raw "closure" pointer.
//!
//! Calling [`GpuFutureCompletion::into_raw`] returns a raw opaque
//! pointer suitable for use as the "closure" pointer in `wgpu_core`'s
//! callbacks. The [`GpuFutureCompletion::from_raw`] converts such a
//! raw opaque pointer back into a [`GpuFutureCompletion`]. See the
//! direct back end's implementation of [`Context::buffer_map_async`]
//! for an example of this.
//!
//! [`complete(v)`]: GpuFutureCompletion::complete
//! [`Waker`]: std::task::Waker
//! [`Future::poll`]: std::future::Future::poll
//! [`wgpu_core::resource::BufferMapOperation`]: https://docs.rs/wgpu-core/latest/wgpu_core/resource/struct.BufferMapOperation.html
//! [`wgpu_core::device::queue::SubmittedWorkDoneClosure`]: https://docs.rs/wgpu-core/latest/wgpu_core/device/queue/struct.SubmittedWorkDoneClosure.html
//! [`Context::buffer_map_async`]: crate::Context::buffer_map_async
use parking_lot::Mutex;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

/// The current state of a `GpuFuture`.
enum WakerOrResult<T> {
    /// The last [`Waker`] used to poll this future, if any.
    ///
    /// [`Waker`]: std::task::Waker
    Waker(Waker),

    /// The value this future resolves to, if it is ready.
    Result(T),
}

/// The shared state of a [`GpuFuture`] and its [`GpuFutureCompletion`].
///
/// Polling the future when it is not yet ready stores the [`Waker`]
/// here; completing the future when it has not yet been polled stores
/// the value here. See [`WakerOrResult`] for details.
type GpuFutureData<T> = Mutex<Option<WakerOrResult<T>>>;

/// A [`Future`] that will be ready when some sort of GPU activity has finished.
///
/// Call [`new_gpu_future`] to create a `GpuFuture`, along with a
/// paired `GpuFutureCompletion` that can be used to mark it as ready.
pub struct GpuFuture<T> {
    data: Arc<GpuFutureData<T>>,
}

/// An opaque type used for pointers to a [`GpuFutureCompletion`]'s guts.
pub enum OpaqueData {}

//TODO: merge this with `GpuFuture` and avoid `Arc` on the data.
/// A completion handle to set the result on a [`GpuFuture`].
pub struct GpuFutureCompletion<T> {
    data: Arc<GpuFutureData<T>>,
}

impl<T> Future for GpuFuture<T> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, context: &mut Context) -> Poll<Self::Output> {
        let mut waker_or_result = self.into_ref().get_ref().data.lock();

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
    /// Mark our paired [`GpuFuture`] as ready, with the given `value`.
    pub fn complete(self, value: T) {
        let mut waker_or_result = self.data.lock();

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

    /// Convert this `GpuFutureCompletion` into a raw pointer for `wgpu_core` to hold.
    pub(crate) fn into_raw(self) -> *mut OpaqueData {
        Arc::into_raw(self.data) as _
    }

    /// Convert a raw pointer returned by [`into_raw`] back into a `GpuFutureCompletion`.
    ///
    /// [`into_raw`]: GpuFutureCompletion::into_raw
    pub(crate) unsafe fn from_raw(this: *mut OpaqueData) -> Self {
        Self {
            data: Arc::from_raw(this as _),
        }
    }
}

/// Construct a fresh [`GpuFuture`] and a paired [`GpuFutureCompletion`].
///
/// See the module docs for details.
pub(crate) fn new_gpu_future<T>() -> (GpuFuture<T>, GpuFutureCompletion<T>) {
    let data = Arc::new(Mutex::new(None));
    (
        GpuFuture {
            data: Arc::clone(&data),
        },
        GpuFutureCompletion { data },
    )
}

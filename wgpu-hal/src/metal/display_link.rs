//! FIFO drawable delivery on Metal's display schedule, independent of main-thread work.

mod queue;
use alloc::sync::Arc;
use core::{
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};
use objc2::{
    available, define_class, msg_send,
    rc::{autoreleasepool, Retained, Weak},
    runtime::{AnyObject, NSObject, ProtocolObject},
    sel, AnyThread, DefinedClass, MainThreadMarker,
};
use objc2_core_foundation::{kCFRunLoopDefaultMode, CFRetained, CFRunLoop};
use objc2_foundation::{
    ns_string, NSNotification, NSNotificationCenter, NSObjectProtocol, NSQualityOfService,
    NSRunLoop, NSRunLoopCommonModes, NSThread,
};
use objc2_quartz_core::{
    CACurrentMediaTime, CAFrameRateRange, CAMetalDisplayLink, CAMetalDisplayLinkDelegate,
    CAMetalDisplayLinkUpdate, CAMetalDrawable, CAMetalLayer,
};
use queue::{Frame, Inbox};
use std::{thread::JoinHandle, time::Instant};
use wgpu_sync::{Condvar, CondvarMutex, Mutex};

/// Whether FIFO surfaces configured on the main thread receive drawables from a display link.
pub(super) fn supported() -> bool {
    available!(macos = 14.0)
}

#[derive(Debug)]
pub(super) struct Drawable(pub Retained<ProtocolObject<dyn CAMetalDrawable>>);
// SAFETY: Metal drawables may be retained and transferred between encoding and presentation threads.
unsafe impl Send for Drawable {}

struct Layer(Retained<CAMetalLayer>);
// SAFETY: only Metal's display-link API uses this retained layer on the worker; AppKit stays on main.
unsafe impl Send for Layer {}

#[derive(Debug)]
struct Wake(CFRetained<CFRunLoop>);
// SAFETY: this handle is used only to stop/wake its run loop, which Core Foundation permits cross-thread.
unsafe impl Send for Wake {}
unsafe impl Sync for Wake {}

/// The latest display rate awaiting the link's thread; zero bits mean none is pending.
#[derive(Debug, Default)]
struct RateRequest(AtomicU32);

impl RateRequest {
    fn request(&self, hz: f32) {
        self.0.store(hz.to_bits(), Ordering::Release);
    }

    fn take(&self) -> Option<f32> {
        let bits = self.0.swap(0, Ordering::Acquire);
        (bits != 0).then(|| f32::from_bits(bits))
    }
}

/// A screen's reported maximum refresh as the link's cadence; unknown screens report zero.
fn frame_rate(maximum_frames_per_second: isize) -> Option<f32> {
    (maximum_frames_per_second > 0).then_some(maximum_frames_per_second as f32)
}

/// Pins the link to `hz`, so the system cannot pick a lower cadence for a FIFO surface.
fn exact_range(hz: f32) -> CAFrameRateRange {
    CAFrameRateRange {
        minimum: hz,
        maximum: hz,
        preferred: hz,
    }
}

/// The native render budget for a surface's configured latency; the link accepts one or two.
fn preferred_frame_latency(maximum_frame_latency: u32) -> f32 {
    maximum_frame_latency.clamp(1, 2) as f32
}

/// The maximum refresh of the screen hosting `window`; AppKit requires the main thread.
fn screen_rate(window: &NSObject, _main: MainThreadMarker) -> Option<f32> {
    // SAFETY: `window` is an `NSWindow`, and both properties are read on the main thread.
    let screen: Option<Retained<NSObject>> = unsafe { msg_send![window, screen] };
    frame_rate(unsafe { msg_send![&*screen?, maximumFramesPerSecond] })
}

#[derive(Debug, Default)]
pub(super) struct Receiver {
    inbox: CondvarMutex<Inbox<Drawable>>,
    ready: Condvar,
    wake: Mutex<Option<Wake>>,
    rate: RateRequest,
}

impl Receiver {
    /// Waits for a display update whose submission deadline has not passed, releasing the lock
    /// while callbacks publish.
    pub fn take(&self, timeout: Duration) -> Option<Retained<ProtocolObject<dyn CAMetalDrawable>>> {
        let until = Instant::now() + timeout;
        let mut inbox = self.inbox.lock();
        loop {
            match inbox.take(CACurrentMediaTime()) {
                Some(Ok(drawable)) => return Some(drawable.0),
                Some(Err(expired)) => {
                    drop(inbox);
                    drop(expired);
                    inbox = self.inbox.lock();
                    continue;
                }
                None if inbox.is_stopped() => return None,
                None => {}
            }
            let remaining = until.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return None;
            }
            self.ready.wait_for(&mut inbox, remaining);
        }
    }

    /// Cancels acquisition and releases prefetched handles before surface reconfiguration.
    fn stop(&self) {
        let discarded = self.inbox.lock().stop();
        self.ready.notify_all();
        drop(discarded);
        if let Some(wake) = &*self.wake.lock() {
            wake.0.stop();
            wake.0.wake_up();
        }
    }
}

define_class!(
    // SAFETY: NSObject has no subclass requirements; the delegate owns its shared state.
    #[unsafe(super(NSObject))]
    #[thread_kind = AnyThread]
    #[ivars = Arc<Receiver>]
    struct Target;
    unsafe impl NSObjectProtocol for Target {}
    unsafe impl CAMetalDisplayLinkDelegate for Target {
        #[unsafe(method(metalDisplayLink:needsUpdate:))]
        fn display_update(&self, link: &CAMetalDisplayLink, update: &CAMetalDisplayLinkUpdate) {
            let receiver = self.ivars();
            if let Some(hz) = receiver.rate.take() {
                link.setPreferredFrameRateRange(exact_range(hz));
            }
            let discarded = receiver.inbox.lock().publish(Frame {
                drawable: Drawable(update.drawable()),
                deadline: update.targetTimestamp(),
            });
            receiver.ready.notify_one();
            drop(discarded);
        }
    }
);

impl Target {
    fn new(receiver: Arc<Receiver>) -> Retained<Self> {
        let object = Self::alloc().set_ivars(receiver);
        // SAFETY: NSObject initialization accepts the initialized delegate allocation.
        unsafe { msg_send![super(object), init] }
    }
}

struct ScreenState {
    receiver: Arc<Receiver>,
    window: Weak<NSObject>,
}

define_class!(
    // SAFETY: NSObject has no subclass requirements; AppKit posts screen changes on main.
    #[unsafe(super(NSObject))]
    #[thread_kind = AnyThread]
    #[ivars = ScreenState]
    struct ScreenObserver;
    unsafe impl NSObjectProtocol for ScreenObserver {}
    impl ScreenObserver {
        /// Requests the new hosting screen's rate after a move or a display mode change.
        #[unsafe(method(screenChanged:))]
        fn screen_changed(&self, _notification: &NSNotification) {
            let state = self.ivars();
            let Some(main) = MainThreadMarker::new() else {
                return;
            };
            if let Some(hz) = state.window.load().and_then(|window| screen_rate(&window, main)) {
                state.receiver.rate.request(hz);
            }
        }
    }
);

/// A notification registration, removed on drop; removal is safe from any thread.
struct Observer(Retained<ScreenObserver>);

impl core::fmt::Debug for Observer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("Observer")
    }
}

impl Observer {
    /// Follows `window` across screens and display mode changes.
    fn new(receiver: Arc<Receiver>, window: &NSObject) -> Self {
        let state = ScreenState {
            receiver,
            window: Weak::new(window),
        };
        let object = ScreenObserver::alloc().set_ivars(state);
        // SAFETY: NSObject initialization accepts the initialized observer allocation.
        let observer: Retained<ScreenObserver> = unsafe { msg_send![super(object), init] };
        let center = NSNotificationCenter::defaultCenter();
        let window: &AnyObject = window;
        // AppKit notification names are string constants equal to their identifiers.
        let registrations = [
            (
                ns_string!("NSWindowDidChangeScreenNotification"),
                Some(window),
            ),
            (
                ns_string!("NSApplicationDidChangeScreenParametersNotification"),
                None,
            ),
        ];
        for (name, object) in registrations {
            // SAFETY: the selector takes one notification argument; `Drop` removes the observer.
            unsafe {
                center.addObserver_selector_name_object(
                    &observer,
                    sel!(screenChanged:),
                    Some(name),
                    object,
                );
            }
        }
        Self(observer)
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        // SAFETY: removes every registration made for this observer in `new`.
        unsafe { NSNotificationCenter::defaultCenter().removeObserver(&self.0) };
    }
}

#[derive(Debug)]
pub(super) struct DisplayLink {
    pub receiver: Arc<Receiver>,
    worker: Option<JoinHandle<()>>,
    _observer: Observer,
}

impl DisplayLink {
    /// Starts native callbacks at the hosting screen's maximum rate, following screen changes.
    pub fn new(
        layer: Retained<CAMetalLayer>,
        window: &NSObject,
        maximum_frame_latency: u32,
        main: MainThreadMarker,
    ) -> Option<Self> {
        let receiver = Arc::new(Receiver::default());
        if let Some(hz) = screen_rate(window, main) {
            receiver.rate.request(hz);
        }
        let observer = Observer::new(receiver.clone(), window);
        let shared = receiver.clone();
        let layer = Layer(layer);
        let latency = preferred_frame_latency(maximum_frame_latency);
        let worker = std::thread::Builder::new()
            .name("Metal display link".into())
            .spawn(move || {
                autoreleasepool(|_| {
                    NSThread::currentThread()
                        .setQualityOfService(NSQualityOfService::UserInteractive);
                    // Keep the wrapper intact so the closure transfers its Send owner, not the raw field.
                    run(layer, shared, latency);
                });
            })
            .ok()?;
        Some(Self {
            receiver,
            worker: Some(worker),
            _observer: observer,
        })
    }
}

/// Owns the native link and registration on one run loop until cancellation.
fn run(layer: Layer, receiver: Arc<Receiver>, latency: f32) {
    let target = Target::new(receiver.clone());
    let link = CAMetalDisplayLink::initWithMetalLayer(CAMetalDisplayLink::alloc(), &layer.0);
    link.setDelegate(Some(ProtocolObject::from_ref(&*target)));
    link.setPreferredFrameLatency(latency);
    if let Some(hz) = receiver.rate.take() {
        link.setPreferredFrameRateRange(exact_range(hz));
    }
    *receiver.wake.lock() = CFRunLoop::current().map(Wake);
    // SAFETY: creation, registration, callbacks and invalidation share this worker's run loop.
    unsafe {
        link.addToRunLoop_forMode(&NSRunLoop::currentRunLoop(), NSRunLoopCommonModes);
    }
    while !receiver.inbox.lock().is_stopped() {
        autoreleasepool(|_| {
            // SAFETY: the constant is supplied by Core Foundation; the worker runs its own loop.
            unsafe {
                CFRunLoop::run_in_mode(kCFRunLoopDefaultMode, 0.1, true);
            }
        });
    }
    link.invalidate();
    link.setDelegate(None);
    receiver.wake.lock().take();
}

impl Drop for DisplayLink {
    fn drop(&mut self) {
        self.receiver.stop();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_rates_apply_once_and_the_latest_change_wins() {
        let rate = RateRequest::default();
        assert_eq!(rate.take(), None);
        rate.request(60.0);
        rate.request(120.0);
        assert_eq!(rate.take(), Some(120.0));
        assert_eq!(
            rate.take(),
            None,
            "an applied rate is not reapplied every update"
        );
    }

    #[test]
    fn screens_without_a_reported_rate_keep_the_current_cadence() {
        assert_eq!(frame_rate(0), None);
        assert_eq!(frame_rate(-1), None);
        assert_eq!(frame_rate(120), Some(120.0));
        let range = exact_range(120.0);
        assert_eq!(
            (range.minimum, range.maximum, range.preferred),
            (120.0, 120.0, 120.0)
        );
    }

    #[test]
    fn configured_latency_sets_the_native_render_budget() {
        assert_eq!(
            preferred_frame_latency(1),
            1.0,
            "one-frame latency is honoured"
        );
        assert_eq!(preferred_frame_latency(2), 2.0);
        assert_eq!(preferred_frame_latency(0), 1.0);
        assert_eq!(preferred_frame_latency(3), 2.0);
    }
}

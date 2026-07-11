//! macOS: observe `NSApplicationDidChangeScreenParametersNotification` so the
//! example re-queries the display's HDR info when the screen configuration
//! changes. winit doesn't surface this event.

use core::ptr::NonNull;

use objc2_foundation::{NSNotification, NSNotificationCenter, NSOperationQueue, NSString};
use winit::event_loop::EventLoopProxy;

use crate::UserEvent;

/// The opaque token returned when registering a notification observer; held for
/// as long as the observer should stay live.
pub(crate) type ScreenObserver =
    objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2::runtime::NSObjectProtocol>>;

/// Register an observer for `NSApplicationDidChangeScreenParametersNotification`
/// that bounces a [`UserEvent::ScreenParametersChanged`] back through the event
/// loop, so the example re-queries the display's HDR info reactively.
///
/// This is Apple's documented way to track EDR headroom: AppKit posts this
/// notification when the display configuration changes — resolution, arrangement,
/// HDR mode toggled in System Settings, the window moving to another display, and
/// the SDR brightness slider, which shifts SDR white and so changes the available
/// EDR headroom. The handler just re-reads `NSScreen`'s EDR values (what
/// `Surface::display_hdr_info` does on the Metal backend). winit doesn't expose
/// this event, so we register our own observer; observers are additive, so this
/// doesn't disturb winit's own.
///
/// The returned token must be kept alive (here, in `App::screen_observer`) for the
/// observer to stay registered.
///
/// Note: ambient-light/auto-brightness drift and the gradual EDR ramp after an
/// EDR layer first appears are continuous and may not post a notification for
/// every step; an app that wants frame-accurate headroom re-reads it each frame
/// in its render loop. The notification covers the discrete changes a user makes.
pub(crate) fn observe_screen_parameter_changes(proxy: EventLoopProxy<UserEvent>) -> ScreenObserver {
    // `NSApplicationDidChangeScreenParametersNotification` is a documented AppKit
    // constant whose string value is the symbol name itself, so we build the
    // `NSString` directly rather than pull in objc2-app-kit for one name.
    let name = NSString::from_str("NSApplicationDidChangeScreenParametersNotification");

    let block = block2::RcBlock::new(move |_note: NonNull<NSNotification>| {
        // Delivered on the main thread (we pass the main queue below), so it is
        // safe here to touch AppKit. Keep it cheap: just wake the loop; the
        // re-query runs in `user_event`.
        let _ = proxy.send_event(UserEvent::ScreenParametersChanged);
    });

    let center = NSNotificationCenter::defaultCenter();
    let main_queue = NSOperationQueue::mainQueue();
    // SAFETY: `name` and `main_queue` are valid for the call; the block is
    // retained by the returned observer token, which the caller keeps alive.
    unsafe {
        center.addObserverForName_object_queue_usingBlock(
            Some(&name),
            None,
            Some(&main_queue),
            &block,
        )
    }
}

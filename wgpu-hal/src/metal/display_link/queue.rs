//! One pending display update; replacing stale work cannot grow presentation latency.

#[derive(Debug)]
pub(super) struct Frame<T> {
    pub drawable: T,
    /// Host time by which the frame must be submitted to make its presentation slot.
    pub deadline: f64,
}

#[derive(Debug)]
pub(super) struct Inbox<T> {
    pending: Option<Frame<T>>,
    stopped: bool,
    requested: bool,
}

impl<T> Default for Inbox<T> {
    fn default() -> Self {
        Self {
            pending: None,
            stopped: false,
            requested: false,
        }
    }
}

impl<T> Inbox<T> {
    /// Returns displaced ownership so native handles are released outside the inbox lock.
    pub fn publish(&mut self, frame: Frame<T>) -> Option<Frame<T>> {
        if self.stopped || !self.requested {
            Some(frame)
        } else {
            self.requested = false;
            self.pending.replace(frame)
        }
    }

    /// Takes the pending drawable unless `now` is past its deadline; an expired one comes back
    /// as `Err` for release outside the lock.
    pub fn take(&mut self, now: f64) -> Option<Result<T, T>> {
        self.pending.take().map(|frame| {
            if now <= frame.deadline {
                Ok(frame.drawable)
            } else {
                Err(frame.drawable)
            }
        })
    }

    /// Requests one update while acquisition is waiting for a drawable.
    pub fn request(&mut self) {
        self.requested = !self.stopped;
    }

    /// Cancels demand when acquisition times out.
    pub fn cancel_request(&mut self) {
        self.requested = false;
    }

    /// Whether callbacks are needed to satisfy an acquisition.
    pub fn needs_update(&self) -> bool {
        self.requested
    }

    pub fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Rejects late callbacks and returns any prefetched drawable for release.
    pub fn stop(&mut self) -> Option<Frame<T>> {
        self.stopped = true;
        self.requested = false;
        self.pending.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds an update with a synthetic submission deadline.
    fn frame(drawable: u32, deadline: f64) -> Frame<u32> {
        Frame { drawable, deadline }
    }

    #[test]
    fn callbacks_are_needed_only_until_requested_work_arrives() {
        let mut inbox = Inbox::default();
        assert!(!inbox.needs_update());
        assert_eq!(inbox.publish(frame(0, 10.0)).unwrap().drawable, 0);
        inbox.request();
        assert!(inbox.needs_update());
        assert!(inbox.publish(frame(1, 10.0)).is_none());
        assert!(!inbox.needs_update());
        assert_eq!(inbox.publish(frame(2, 10.0)).unwrap().drawable, 2);
        assert_eq!(inbox.take(0.0), Some(Ok(1)));
        inbox.request();
        assert!(inbox.needs_update());
        inbox.cancel_request();
        assert!(!inbox.needs_update());
        assert_eq!(inbox.publish(frame(3, 10.0)).unwrap().drawable, 3);
        inbox.stop();
        inbox.request();
        assert!(!inbox.needs_update());
    }

    #[test]
    fn expired_updates_need_another_request() {
        let mut inbox = Inbox::default();
        inbox.request();
        inbox.publish(frame(1, 10.0));
        assert_eq!(inbox.take(11.0), Some(Err(1)));
        assert!(!inbox.needs_update());
        inbox.request();
        assert!(inbox.needs_update());
        assert!(inbox.publish(frame(2, 20.0)).is_none());
        assert_eq!(inbox.take(12.0), Some(Ok(2)));
        assert!(!inbox.needs_update());
    }

    #[test]
    fn slow_consumers_receive_the_latest_update_without_a_backlog() {
        let mut inbox = Inbox::default();
        inbox.request();
        assert!(inbox.publish(frame(1, 10.0)).is_none());
        inbox.request();
        assert_eq!(inbox.publish(frame(2, 20.0)).unwrap().drawable, 1);
        assert_eq!(inbox.take(15.0), Some(Ok(2)));
        assert_eq!(inbox.take(15.0), None);
    }

    #[test]
    fn updates_past_their_submission_deadline_are_released_not_rendered() {
        let mut inbox = Inbox::default();
        inbox.request();
        inbox.publish(frame(1, 10.0));
        assert_eq!(
            inbox.take(10.0),
            Some(Ok(1)),
            "the deadline itself is on time"
        );
        inbox.request();
        inbox.publish(frame(2, 10.0));
        assert_eq!(inbox.take(10.5), Some(Err(2)));
        assert_eq!(inbox.take(10.5), None, "an expired update is not retried");
    }

    #[test]
    fn stopping_releases_prefetch_and_rejects_late_callbacks() {
        let mut inbox = Inbox::default();
        inbox.request();
        inbox.publish(frame(1, 10.0));
        assert_eq!(inbox.stop().unwrap().drawable, 1);
        assert!(inbox.is_stopped());
        assert_eq!(inbox.publish(frame(2, 20.0)).unwrap().drawable, 2);
        assert_eq!(inbox.take(0.0), None);
        assert!(inbox.stop().is_none());
    }
}

use core::time::Duration;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};

#[derive(Debug, Default)]
pub(super) struct Completion {
    value: Mutex<crate::FenceValue>,
    changed: Condvar,
}

impl Completion {
    pub(super) fn value(&self) -> crate::FenceValue {
        *self.value.lock()
    }

    pub(super) fn publish(&self, value: crate::FenceValue, succeeded: bool) -> crate::FenceValue {
        let mut completed = self.value.lock();
        // Native callbacks may arrive out of order. Failed buffers only wake
        // waiters so they can report the native error, not successful completion.
        if succeeded && value > *completed {
            *completed = value;
            self.changed.notify_all();
        } else if !succeeded {
            self.changed.notify_all();
        }
        *completed
    }

    pub(super) fn wait(
        &self,
        value: crate::FenceValue,
        timeout: Option<Duration>,
        mut native_status: impl FnMut() -> Result<bool, crate::DeviceError>,
    ) -> Result<bool, crate::DeviceError> {
        // Represent the fixed deadline as origin + duration to avoid overflowing
        // Instant for large timeouts. Wakeups never reset this origin.
        let start = Instant::now();
        let mut completed = self.value.lock();
        loop {
            if native_status()? || *completed >= value {
                return Ok(true);
            }
            match timeout {
                Some(timeout) => {
                    let remaining = timeout.saturating_sub(start.elapsed());
                    if remaining.is_zero() {
                        return Ok(false);
                    }
                    self.changed.wait_for(&mut completed, remaining);
                }
                None => self.changed.wait(&mut completed),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::sync::Arc;
    use std::{
        sync::{mpsc, Barrier},
        thread,
    };

    const BOUND: Duration = Duration::from_secs(5);

    #[test]
    fn cpu_early_completion_and_zero_timeout() {
        let completion = Completion::default();
        assert!(!completion
            .wait(1, Some(Duration::ZERO), || Ok(false))
            .unwrap());
        completion.publish(1, true);
        assert!(completion
            .wait(1, Some(Duration::ZERO), || Ok(false))
            .unwrap());
        assert!(completion.wait(1, None, || Ok(false)).unwrap());
    }

    #[test]
    fn cpu_native_status_shortcuts() {
        let completion = Completion::default();
        assert!(completion
            .wait(1, Some(Duration::ZERO), || Ok(true))
            .unwrap());
        assert!(matches!(
            completion.wait(1, None, || Err(crate::DeviceError::Lost)),
            Err(crate::DeviceError::Lost)
        ));
    }

    #[test]
    fn cpu_check_to_wait_handoff() {
        let completion = Completion::default();
        let handoff = Barrier::new(2);
        thread::scope(|scope| {
            let publisher = scope.spawn(|| {
                handoff.wait();
                // The waiter still owns the predicate lock at the barrier.
                completion.publish(1, true);
            });
            let mut first = true;
            assert!(completion
                .wait(1, Some(BOUND), || {
                    if first {
                        first = false;
                        handoff.wait();
                    }
                    Ok(false)
                })
                .unwrap());
            publisher.join().unwrap();
        });
    }

    #[test]
    fn cpu_completion_during_lock_acquisition() {
        let completion = Completion::default();
        let mut guard = completion.value.lock();
        let (started, ready) = mpsc::channel();
        thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                started.send(()).unwrap();
                completion.wait(1, Some(BOUND), || Ok(false)).unwrap()
            });
            ready.recv_timeout(BOUND).unwrap();
            *guard = 1;
            completion.changed.notify_all();
            drop(guard);
            assert!(waiter.join().unwrap());
        });
    }

    #[test]
    fn cpu_indefinite_wait_and_multiple_targets() {
        let completion = Completion::default();
        let (checked, checks) = mpsc::channel();
        let (finished, finishes) = mpsc::channel();
        thread::scope(|scope| {
            for target in [1, 2, 2] {
                let checked = checked.clone();
                let finished = finished.clone();
                let completion = &completion;
                scope.spawn(move || {
                    let result = completion.wait(target, None, || {
                        checked.send(target).unwrap();
                        Ok(false)
                    });
                    finished.send((target, result.unwrap())).unwrap();
                });
            }
            for _ in 0..3 {
                checks.recv_timeout(BOUND).unwrap();
            }
            completion.publish(1, true);
            let first = finishes.recv_timeout(BOUND);
            assert!(finishes.try_recv().is_err());
            // Release all waiters even if the first assertion below fails.
            completion.publish(2, true);
            assert_eq!(first.unwrap(), (1, true));
            for _ in 0..2 {
                assert_eq!(finishes.recv_timeout(BOUND).unwrap(), (2, true));
            }
        });
    }

    #[test]
    fn cpu_irrelevant_and_spurious_wakeups_keep_deadline() {
        let completion = Completion::default();
        let (checked, checks) = mpsc::channel();
        let (finished, finishes) = mpsc::channel();
        let (stop, stopped) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| {
                let start = Instant::now();
                let result = completion.wait(100, Some(Duration::from_millis(200)), || {
                    checked.send(()).unwrap();
                    Ok(false)
                });
                finished.send((result.unwrap(), start.elapsed())).unwrap();
            });
            checks.recv_timeout(BOUND).unwrap();
            let completion = &completion;
            scope.spawn(move || {
                while matches!(
                    stopped.recv_timeout(Duration::from_millis(10)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                ) {
                    completion.publish(1, true);
                    completion.changed.notify_all();
                }
            });
            let result = finishes.recv_timeout(BOUND);
            stop.send(()).unwrap();
            let (ready, elapsed) = result.unwrap();
            assert!(!ready);
            assert!(elapsed >= Duration::from_millis(200));
            assert!(elapsed < BOUND);
            assert!(checks.try_iter().count() > 1);
        });
    }

    #[test]
    fn cpu_ready_status_wins_at_timeout_boundary() {
        let completion = Completion::default();
        let (_send, receive) = mpsc::channel::<()>();
        assert!(completion
            .wait(1, Some(Duration::from_millis(10)), || {
                assert_eq!(
                    receive.recv_timeout(Duration::from_millis(20)),
                    Err(mpsc::RecvTimeoutError::Timeout)
                );
                Ok(true)
            })
            .unwrap());
    }

    #[test]
    fn cpu_submillisecond_timeout() {
        let completion = Completion::default();
        let timeout = Duration::from_micros(100);
        let start = Instant::now();
        assert!(!completion.wait(1, Some(timeout), || Ok(false)).unwrap());
        assert!(start.elapsed() >= timeout);
        assert!(start.elapsed() < BOUND);
    }

    #[test]
    fn cpu_large_timeout_wakes_without_overflow() {
        let completion = Completion::default();
        let handoff = Barrier::new(2);
        thread::scope(|scope| {
            scope.spawn(|| {
                handoff.wait();
                completion.publish(1, true);
            });
            let mut first = true;
            assert!(completion
                .wait(1, Some(Duration::MAX), || {
                    if first {
                        first = false;
                        handoff.wait();
                    }
                    Ok(false)
                })
                .unwrap());
        });
    }

    #[test]
    fn cpu_publication_is_monotonic() {
        let completion = Completion::default();
        for value in [3, 1, 2, 3, 0] {
            completion.publish(value, true);
            assert_eq!(completion.value(), 3);
        }
        completion.publish(4, false);
        assert_eq!(completion.value(), 3);
        assert!(completion
            .wait(3, Some(Duration::ZERO), || Ok(false))
            .unwrap());
    }

    #[test]
    fn cpu_failed_completion_wakes_native_error() {
        let completion = Completion::default();
        let (checked, checks) = mpsc::channel();
        thread::scope(|scope| {
            let completion = &completion;
            scope.spawn(move || {
                checks.recv_timeout(BOUND).unwrap();
                completion.publish(1, false);
            });
            let mut first = true;
            assert!(matches!(
                completion.wait(1, Some(BOUND), || {
                    if first {
                        first = false;
                        checked.send(()).unwrap();
                        Ok(false)
                    } else {
                        Err(crate::DeviceError::Lost)
                    }
                }),
                Err(crate::DeviceError::Lost)
            ));
        });
        assert_eq!(completion.value(), 0);
    }

    #[test]
    fn cpu_completed_block_keeps_state_alive() {
        let completion = Arc::new(Completion::default());
        let weak = Arc::downgrade(&completion);
        let captured = Arc::clone(&completion);
        let block = block2::RcBlock::new(move || captured.publish(1, true));
        drop(completion);
        block.call(());
        assert_eq!(weak.upgrade().unwrap().value(), 1);
        drop(block);
        assert!(weak.upgrade().is_none());
    }
}

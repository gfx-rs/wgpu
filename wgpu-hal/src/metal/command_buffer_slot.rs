use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Admission held from before native allocation until commit or native buffer drop.
#[derive(Debug)]
pub(super) struct CommandBufferSlot {
    count: Arc<AtomicUsize>,
    submitted: AtomicBool,
}

impl CommandBufferSlot {
    pub(super) fn reserve(count: &Arc<AtomicUsize>, limit: usize) -> Result<Self, usize> {
        count.fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
            (count < limit).then_some(count + 1)
        })?;
        Ok(Self {
            count: Arc::clone(count),
            submitted: AtomicBool::new(false),
        })
    }

    pub(super) fn mark_submitted(&self) {
        assert!(!self.submitted.swap(true, Ordering::AcqRel));
        self.release();
    }

    fn release(&self) {
        let previous = self.count.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
    }
}

impl Drop for CommandBufferSlot {
    fn drop(&mut self) {
        if !*self.submitted.get_mut() {
            self.release();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn cpu_admission_refusal_and_reuse() {
        let count = Arc::new(AtomicUsize::new(0));
        let first = CommandBufferSlot::reserve(&count, 3).unwrap();
        assert_eq!(count.load(Ordering::Acquire), 1);
        let second = CommandBufferSlot::reserve(&count, 3).unwrap();
        assert_eq!(count.load(Ordering::Acquire), 2);
        let third = CommandBufferSlot::reserve(&count, 3).unwrap();
        assert_eq!(count.load(Ordering::Acquire), 3);
        for _ in 0..2 {
            assert_eq!(CommandBufferSlot::reserve(&count, 3).unwrap_err(), 3);
            assert_eq!(count.load(Ordering::Acquire), 3);
        }
        drop(second);
        assert_eq!(count.load(Ordering::Acquire), 2);
        let replacement = CommandBufferSlot::reserve(&count, 3).unwrap();
        assert_eq!(count.load(Ordering::Acquire), 3);
        drop((first, third, replacement));
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert_eq!(Arc::strong_count(&count), 1);
    }

    #[test]
    fn cpu_discard_clears_encoder_reservation() {
        let count = Arc::new(AtomicUsize::new(0));
        let mut encoder_slot = Some(CommandBufferSlot::reserve(&count, 1).unwrap());
        assert!(encoder_slot.is_some());
        // discard_encoding clears the raw buffer before clearing this Option.
        encoder_slot = None;
        assert_eq!(count.load(Ordering::Acquire), 0);
        drop(encoder_slot.take());
        assert_eq!(count.load(Ordering::Acquire), 0);
        encoder_slot = Some(CommandBufferSlot::reserve(&count, 1).unwrap());
        drop(encoder_slot);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn cpu_finish_transfers_reservation_until_buffer_drop() {
        let count = Arc::new(AtomicUsize::new(0));
        let mut encoder_slot = Some(CommandBufferSlot::reserve(&count, 2).unwrap());
        // end_encoding moves the slot without releasing it; reset_all drops buffers.
        let finished_slot = encoder_slot.take().unwrap();
        drop(encoder_slot);
        assert_eq!(count.load(Ordering::Acquire), 1);
        let next_recording = CommandBufferSlot::reserve(&count, 2).unwrap();
        drop(finished_slot);
        assert_eq!(count.load(Ordering::Acquire), 1);
        drop(next_recording);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn cpu_submit_releases_before_buffer_drop_once() {
        let count = Arc::new(AtomicUsize::new(0));
        let submitted = CommandBufferSlot::reserve(&count, 1).unwrap();
        // Queue::submit calls this after raw.commit(), not at finish or completion.
        submitted.mark_submitted();
        assert_eq!(count.load(Ordering::Acquire), 0);
        let next_recording = CommandBufferSlot::reserve(&count, 1).unwrap();
        drop(submitted);
        assert_eq!(count.load(Ordering::Acquire), 1);
        drop(next_recording);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn cpu_duplicate_submission_does_not_double_release() {
        let count = Arc::new(AtomicUsize::new(0));
        let submitted = CommandBufferSlot::reserve(&count, 1).unwrap();
        submitted.mark_submitted();
        let next_recording = CommandBufferSlot::reserve(&count, 1).unwrap();
        assert!(std::panic::catch_unwind(|| submitted.mark_submitted()).is_err());
        assert_eq!(count.load(Ordering::Acquire), 1);
        drop(submitted);
        assert_eq!(count.load(Ordering::Acquire), 1);
        drop(next_recording);
        assert_eq!(count.load(Ordering::Acquire), 0);
    }

    #[test]
    fn cpu_allocation_unwind_releases_reservation() {
        let count = Arc::new(AtomicUsize::new(0));
        let result = std::panic::catch_unwind(|| {
            let _slot = CommandBufferSlot::reserve(&count, 1).unwrap();
            // begin_encoding keeps the reservation local while native allocation runs.
            panic!("simulate allocation failure before storing the raw buffer");
        });
        assert!(result.is_err());
        assert_eq!(count.load(Ordering::Acquire), 0);
        let slot = CommandBufferSlot::reserve(&count, 1).unwrap();
        drop(slot);
        assert_eq!(Arc::strong_count(&count), 1);
    }

    #[test]
    fn cpu_concurrent_admissions_respect_limit() {
        let count = Arc::new(AtomicUsize::new(0));
        let start = Barrier::new(9);
        let admitted = Barrier::new(9);
        let release = Barrier::new(9);
        std::thread::scope(|scope| {
            let mut threads = alloc::vec::Vec::new();
            for _ in 0..8 {
                threads.push(scope.spawn(|| {
                    start.wait();
                    let slot = CommandBufferSlot::reserve(&count, 3);
                    admitted.wait();
                    release.wait();
                    match slot {
                        Ok(slot) => {
                            drop(slot);
                            true
                        }
                        Err(current) => {
                            assert_eq!(current, 3);
                            false
                        }
                    }
                }));
            }
            start.wait();
            admitted.wait();
            let held = count.load(Ordering::Acquire);
            release.wait();
            assert_eq!(held, 3);
            let successes = threads
                .into_iter()
                .map(|thread| usize::from(thread.join().unwrap()))
                .sum::<usize>();
            assert_eq!(successes, 3);
        });
        assert_eq!(count.load(Ordering::Acquire), 0);
        assert_eq!(Arc::strong_count(&count), 1);
    }

    #[test]
    fn cpu_reservation_keeps_counter_alive() {
        let count = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&count);
        let slot = CommandBufferSlot::reserve(&count, 1).unwrap();
        drop(count);
        assert_eq!(weak.upgrade().unwrap().load(Ordering::Acquire), 1);
        slot.mark_submitted();
        assert_eq!(weak.upgrade().unwrap().load(Ordering::Acquire), 0);
        drop(slot);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn cpu_production_capacity_reserves_one_internal_slot() {
        assert_eq!(super::super::adapter::MAX_COMMAND_BUFFERS, 4096);
        assert_eq!(super::super::adapter::MAX_UNSUBMITTED_COMMAND_BUFFERS, 4095);
        assert_eq!(
            super::super::adapter::MAX_COMMAND_BUFFERS
                - super::super::adapter::MAX_UNSUBMITTED_COMMAND_BUFFERS,
            1
        );
    }
}

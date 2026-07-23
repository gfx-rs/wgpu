use core::ops::Range;

use crate::{
    init_tracker::BufferInitTracker,
    resource::{BufferAccessError, BufferAccessResult},
};

pub(super) fn mark_buffer_range_initialized(
    buffer_size: wgt::BufferAddress,
    initialization_status: &mut BufferInitTracker,
    range: Range<wgt::BufferAddress>,
) -> BufferAccessResult {
    if range.start >= range.end {
        return Err(BufferAccessError::InvalidRange {
            start: range.start,
            end: range.end,
        });
    }
    if range.start > buffer_size {
        return Err(BufferAccessError::OutOfBoundsStartOffsetOverrun {
            index: range.start,
            max: buffer_size,
        });
    }
    if range.end > buffer_size {
        return Err(BufferAccessError::OutOfBoundsEndOffsetOverrun {
            index: range.start,
            size: range.end - range.start,
            max: buffer_size,
        });
    }

    drop(initialization_status.drain(range));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::mark_buffer_range_initialized;
    use crate::init_tracker::BufferInitTracker;
    use crate::resource::BufferAccessError;
    use core::ops::Range;

    fn rejected_range(range: Range<u64>) -> BufferAccessError {
        let mut tracker = BufferInitTracker::new(64);
        let error = mark_buffer_range_initialized(64, &mut tracker, range).unwrap_err();
        assert_eq!(tracker.check(0..64), Some(0..64));
        error
    }

    #[test]
    fn marks_only_the_external_write_range_initialized() {
        let mut tracker = BufferInitTracker::new(64);

        mark_buffer_range_initialized(64, &mut tracker, 16..48).unwrap();

        assert_eq!(tracker.check(0..16), Some(0..16));
        assert_eq!(tracker.check(16..48), None);
        assert_eq!(tracker.check(48..64), Some(48..64));
    }

    #[test]
    fn rejects_empty_reversed_and_out_of_bounds_ranges() {
        let reversed = Range { start: 32, end: 16 };
        assert!(matches!(
            rejected_range(16..16),
            BufferAccessError::InvalidRange { start: 16, end: 16 }
        ));
        assert!(matches!(
            rejected_range(reversed),
            BufferAccessError::InvalidRange { start: 32, end: 16 }
        ));
        assert!(matches!(
            rejected_range(65..66),
            BufferAccessError::OutOfBoundsStartOffsetOverrun { index: 65, max: 64 }
        ));
        assert!(matches!(
            rejected_range(48..65),
            BufferAccessError::OutOfBoundsEndOffsetOverrun {
                index: 48,
                size: 17,
                max: 64
            }
        ));
    }
}

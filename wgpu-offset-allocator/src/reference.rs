//! A deliberately naive reference model of a suballocator, used by the differential
//! proptest suite as an oracle.
//!
//! It tracks free ranges as a sorted list and allocates first-fit. It is not
//! performant and makes no claim to match the exact placement of [`Tlsf`] — only the
//! *invariants* (offsets aligned, allocations fit, no overlap, granularity respected,
//! free-size accounting) are asserted against it.
//!
//! [`Tlsf`]: crate::Tlsf

use alloc::vec::Vec;

use crate::math::blocks_on_same_page;
use crate::AllocationType;

/// A live allocation recorded by the model.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ModelAlloc {
    pub offset: u64,
    pub size: u64,
    pub alloc_type: AllocationType,
}

/// The reference model.
#[derive(Debug)]
pub(crate) struct ReferenceModel {
    size: u64,
    granularity: u64,
    /// Live allocations, kept sorted by offset.
    live: Vec<ModelAlloc>,
}

impl ReferenceModel {
    pub(crate) fn new(size: u64, granularity: u64) -> Self {
        ReferenceModel {
            size,
            granularity,
            live: Vec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.live.clear();
    }

    pub(crate) fn allocation_count(&self) -> usize {
        self.live.len()
    }

    /// Sum of free space: total minus used.
    pub(crate) fn sum_free_size(&self) -> u64 {
        let used: u64 = self.live.iter().map(|a| a.size).sum();
        self.size - used
    }

    /// Commits an allocation at `offset`, keeping `live` sorted.
    pub(crate) fn insert(&mut self, offset: u64, size: u64, alloc_type: AllocationType) {
        let pos = self
            .live
            .binary_search_by(|a| a.offset.cmp(&offset))
            .unwrap_or_else(|e| e);
        self.live.insert(
            pos,
            ModelAlloc {
                offset,
                size,
                alloc_type,
            },
        );
    }

    /// Removes the allocation at `offset`.
    pub(crate) fn remove(&mut self, offset: u64) {
        if let Ok(pos) = self.live.binary_search_by(|a| a.offset.cmp(&offset)) {
            self.live.remove(pos);
        }
    }

    /// Checks that a proposed placement does not overlap any live allocation and
    /// respects granularity against neighbours. Used to validate the *real*
    /// allocator's returned offset. Returns `Ok(())` or a description.
    pub(crate) fn check_placement(
        &self,
        offset: u64,
        size: u64,
        alloc_type: AllocationType,
    ) -> Result<(), &'static str> {
        let end = offset.checked_add(size).ok_or("placement overflows u64")?;
        if end > self.size {
            return Err("placement extends past block end");
        }
        for a in &self.live {
            let a_end = a.offset + a.size;
            // Overlap check.
            if offset < a_end && a.offset < end {
                return Err("placement overlaps an existing allocation");
            }
            // Granularity check against neighbours sharing a page.
            if self.granularity > 1 && alloc_type.conflicts_with(a.alloc_type) {
                let (lo, hi) = if offset < a.offset {
                    ((offset, size), (a.offset, a.size))
                } else {
                    ((a.offset, a.size), (offset, size))
                };
                if blocks_on_same_page(lo.0, lo.1, hi.0, self.granularity) {
                    return Err(
                        "placement conflicts with a neighbour on the same granularity page",
                    );
                }
            }
        }
        Ok(())
    }
}

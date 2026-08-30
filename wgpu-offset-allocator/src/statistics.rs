//! Allocation statistics, ported from VMA's `VmaStatistics` / `VmaDetailedStatistics`.
//!
//! These are plain accumulator structs so a higher layer can feed them into its own
//! report type (e.g. wgpu's `AllocatorReport`).

/// Basic usage statistics (`VmaStatistics`).
///
/// Accumulate across blocks with [`add_statistics`](crate::Suballocator::add_statistics).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Statistics {
    /// Number of blocks.
    pub block_count: u32,
    /// Number of live allocations.
    pub allocation_count: u32,
    /// Total size of all blocks.
    pub block_bytes: u64,
    /// Total size occupied by live allocations. Always `<= block_bytes`.
    pub allocation_bytes: u64,
}

impl Statistics {
    /// Merges `src` into `self`.
    pub fn add(&mut self, src: &Statistics) {
        self.block_count += src.block_count;
        self.allocation_count += src.allocation_count;
        self.block_bytes += src.block_bytes;
        self.allocation_bytes += src.allocation_bytes;
    }
}

/// Detailed usage statistics (`VmaDetailedStatistics`): slower to compute, richer.
///
/// The `*_min` fields are [`u64::MAX`] and the `*_max`/counts are `0` when there is
/// nothing of that kind, matching VMA's use of `VK_WHOLE_SIZE` sentinels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DetailedStatistics {
    /// Basic statistics.
    pub statistics: Statistics,
    /// Number of free ranges between allocations.
    pub unused_range_count: u32,
    /// Smallest live allocation size ([`u64::MAX`] if there are none).
    pub allocation_size_min: u64,
    /// Largest live allocation size (`0` if there are none).
    pub allocation_size_max: u64,
    /// Smallest free range size ([`u64::MAX`] if there are none).
    pub unused_range_size_min: u64,
    /// Largest free range size (`0` if there are none).
    pub unused_range_size_max: u64,
}

impl Default for DetailedStatistics {
    fn default() -> Self {
        // Port of VmaClearDetailedStatistics.
        DetailedStatistics {
            statistics: Statistics::default(),
            unused_range_count: 0,
            allocation_size_min: u64::MAX,
            allocation_size_max: 0,
            unused_range_size_min: u64::MAX,
            unused_range_size_max: 0,
        }
    }
}

impl DetailedStatistics {
    /// Records a live allocation of `size` (port of `VmaAddDetailedStatisticsAllocation`).
    pub(crate) fn add_allocation(&mut self, size: u64) {
        self.statistics.allocation_count += 1;
        self.statistics.allocation_bytes += size;
        self.allocation_size_min = self.allocation_size_min.min(size);
        self.allocation_size_max = self.allocation_size_max.max(size);
    }

    /// Records a free range of `size` (port of `VmaAddDetailedStatisticsUnusedRange`).
    pub(crate) fn add_unused_range(&mut self, size: u64) {
        self.unused_range_count += 1;
        self.unused_range_size_min = self.unused_range_size_min.min(size);
        self.unused_range_size_max = self.unused_range_size_max.max(size);
    }
}

//! `VK_EXT_memory_budget` checks mirroring `gpu-allocator` 0.28 memory-type selection
//! (re-verify when upgrading `gpu-allocator`).
//!
//! `heapUsage` is tracked at `VkDeviceMemory` granularity, so `heapUsage + size` does not account
//! for free space inside existing `gpu-allocator` blocks or new block allocations.

use ash::vk;
use gpu_allocator::MemoryLocation;

/// Flags `gpu-allocator` requires after dropping optional preferences (`DEVICE_LOCAL`
/// for `CpuToGpu`, `HOST_CACHED` for `GpuToCpu`).
fn required_flags(location: MemoryLocation) -> vk::MemoryPropertyFlags {
    match location {
        // Transient `LAZILY_ALLOCATED` types always include `DEVICE_LOCAL`.
        MemoryLocation::GpuOnly => vk::MemoryPropertyFlags::DEVICE_LOCAL,
        MemoryLocation::CpuToGpu | MemoryLocation::GpuToCpu => {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        }
        MemoryLocation::Unknown => vk::MemoryPropertyFlags::empty(),
    }
}

pub(super) struct MemoryProperties {
    base: vk::PhysicalDeviceMemoryProperties,
    heap_budget: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
    heap_usage: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
}

impl MemoryProperties {
    pub(super) fn new(
        base: vk::PhysicalDeviceMemoryProperties,
        heap_budget: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
        heap_usage: [vk::DeviceSize; vk::MAX_MEMORY_HEAPS],
    ) -> Self {
        Self {
            base,
            heap_budget,
            heap_usage,
        }
    }

    fn types(&self) -> &[vk::MemoryType] {
        self.base.memory_types_as_slice()
    }

    fn heaps(&self) -> &[vk::MemoryHeap] {
        self.base.memory_heaps_as_slice()
    }

    pub(super) fn heap_budget(&self) -> &[vk::DeviceSize] {
        &self.heap_budget[..self.heaps().len()]
    }

    pub(super) fn heap_usage(&self) -> &[vk::DeviceSize] {
        &self.heap_usage[..self.heaps().len()]
    }

    /// Strict `<` treats a `size == 0` probe at the limit as exhausted (used by
    /// [`Self::exhausted_device_loss_location`]). `VK_EXT_memory_budget` requires non-zero
    /// `heapBudget`; `0` is a driver bug and falls back to `heap.size`.
    fn heap_is_within_budget(
        &self,
        heap_index: usize,
        size: vk::DeviceSize,
        threshold: u8,
    ) -> bool {
        let (Some(heap), Some(&usage), Some(&budget)) = (
            self.heaps().get(heap_index),
            self.heap_usage().get(heap_index),
            self.heap_budget().get(heap_index),
        ) else {
            return false;
        };
        let budget = if budget == 0 { heap.size } else { budget };
        let projected_usage = u128::from(usage) + u128::from(size);
        let limit = u128::from(budget) * u128::from(threshold) / 100;
        projected_usage < limit
    }

    /// Narrows `allowed_bits` to within-budget types for `CpuToGpu`, or returns `allowed_bits`
    /// unchanged for other locations if `gpu-allocator`'s chosen heap fits (`0` otherwise).
    #[must_use]
    pub(super) fn budgeted_memory_type_bits(
        &self,
        location: MemoryLocation,
        allowed_bits: u32,
        size: vk::DeviceSize,
        threshold: u8,
    ) -> u32 {
        let required = required_flags(location);

        // `gpu-allocator` tries BAR first and falls back to host RAM; dropping over-budget types
        // steers both steps. Keeping only `required` types makes exhausted BAR + host yield `0`
        // (`OutOfMemory`) instead of leaving e.g. VRAM set and hitting
        // `NoCompatibleMemoryTypeFound` (`DeviceError::Lost`).
        if location == MemoryLocation::CpuToGpu {
            return self
                .types()
                .iter()
                .enumerate()
                .fold(0, |bits, (index, memory_type)| {
                    let bit = 1 << index;
                    if allowed_bits & bit != 0
                        && memory_type.property_flags.contains(required)
                        && self.heap_is_within_budget(
                            memory_type.heap_index as usize,
                            size,
                            threshold,
                        )
                    {
                        bits | bit
                    } else {
                        bits
                    }
                });
        }

        // Other locations have no fallback on allocation failure, so only the type
        // `gpu-allocator` picks is checked. Narrowing would move `GpuOnly` from exhausted VRAM to
        // BAR (also `DEVICE_LOCAL`) or `GpuToCpu` from cached to uncached memory.
        let find_matching = |flags: vk::MemoryPropertyFlags| {
            self.types()
                .iter()
                .enumerate()
                .find(|&(index, memory_type)| {
                    allowed_bits & (1 << index) != 0 && memory_type.property_flags.contains(flags)
                })
        };

        let preferred_type = if location == MemoryLocation::GpuToCpu {
            find_matching(required | vk::MemoryPropertyFlags::HOST_CACHED)
                .or_else(|| find_matching(required))
        } else {
            find_matching(required)
        };

        let Some((_, preferred_type)) = preferred_type else {
            return 0;
        };

        if self.heap_is_within_budget(preferred_type.heap_index as usize, size, threshold) {
            allowed_bits
        } else {
            0
        }
    }

    /// Returns the first `wgpu-hal` location (`GpuOnly`, `CpuToGpu`, `GpuToCpu`) whose memory has
    /// reached or exceeded `threshold`% of its budget.
    ///
    /// Checking locations rather than heaps keeps the device alive when only BAR is full while
    /// `CpuToGpu` still fits in host memory. `LAZILY_ALLOCATED` types are skipped: they may
    /// precede the regular `DEVICE_LOCAL` type, yet only back transient images, which
    /// `create_texture` budget-checks.
    #[must_use]
    pub(super) fn exhausted_device_loss_location(
        &self,
        valid_memory_types: u32,
        threshold: u8,
    ) -> Option<MemoryLocation> {
        let non_lazy_bits = self
            .types()
            .iter()
            .enumerate()
            .fold(0, |bits, (index, ty)| {
                let bit = 1 << index;
                if valid_memory_types & bit != 0
                    && !ty
                        .property_flags
                        .intersects(vk::MemoryPropertyFlags::LAZILY_ALLOCATED)
                {
                    bits | bit
                } else {
                    bits
                }
            });

        [
            MemoryLocation::GpuOnly,
            MemoryLocation::CpuToGpu,
            MemoryLocation::GpuToCpu,
        ]
        .into_iter()
        .find(|&location| {
            let required = required_flags(location);
            // Distinguish "over budget" from "no such memory class": both return 0 below.
            let has_compatible_type = self.types().iter().enumerate().any(|(index, ty)| {
                non_lazy_bits & (1 << index) != 0 && ty.property_flags.contains(required)
            });
            has_compatible_type
                && self.budgeted_memory_type_bits(location, non_lazy_bits, 0, threshold) == 0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NON_COHERENT_HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_CACHED.as_raw(),
    );
    const HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw()
            | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
    );
    const LOCAL_HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        HOST.as_raw() | vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw(),
    );
    const CACHED_HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        HOST.as_raw() | vk::MemoryPropertyFlags::HOST_CACHED.as_raw(),
    );
    const LOCAL_CACHED_HOST: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        CACHED_HOST.as_raw() | vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw(),
    );
    const LAZY_LOCAL: vk::MemoryPropertyFlags = vk::MemoryPropertyFlags::from_raw(
        vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw()
            | vk::MemoryPropertyFlags::LAZILY_ALLOCATED.as_raw(),
    );
    const MIB: u64 = 1024 * 1024;

    const fn bit(index: u32) -> u32 {
        1 << index
    }

    fn all_types_mask(properties: &MemoryProperties) -> u32 {
        ((1u64 << properties.types().len()) - 1) as u32
    }

    fn matching_types_mask(
        properties: &MemoryProperties,
        predicate: impl Fn(usize, &vk::MemoryType) -> bool,
    ) -> u32 {
        properties
            .types()
            .iter()
            .enumerate()
            .fold(0, |mask, (index, ty)| {
                if predicate(index, ty) {
                    mask | bit(index as u32)
                } else {
                    mask
                }
            })
    }

    /// Each heap entry is `(size, budget, usage)`.
    fn properties_with_sizes(
        types: &[(u32, vk::MemoryPropertyFlags)],
        heaps: &[(u64, u64, u64)],
    ) -> MemoryProperties {
        let mut base = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: types.len() as u32,
            memory_heap_count: heaps.len() as u32,
            ..Default::default()
        };
        for (slot, &(heap_index, property_flags)) in base.memory_types.iter_mut().zip(types) {
            *slot = vk::MemoryType {
                property_flags,
                heap_index,
            };
        }
        let mut heap_budget = [0; vk::MAX_MEMORY_HEAPS];
        let mut heap_usage = [0; vk::MAX_MEMORY_HEAPS];
        for (index, &(size, budget, usage)) in heaps.iter().enumerate() {
            base.memory_heaps[index].size = size;
            heap_budget[index] = budget;
            heap_usage[index] = usage;
        }
        MemoryProperties::new(base, heap_budget, heap_usage)
    }

    fn properties(
        types: &[(u32, vk::MemoryPropertyFlags)],
        budgets_and_usage: &[(u64, u64)],
    ) -> MemoryProperties {
        let mut heaps = [(0, 0, 0); vk::MAX_MEMORY_HEAPS];
        for (slot, &(budget, usage)) in heaps.iter_mut().zip(budgets_and_usage) {
            *slot = (budget.saturating_add(usage), budget, usage);
        }
        properties_with_sizes(types, &heaps[..budgets_and_usage.len()])
    }

    /// Discrete GPU layout without ReBAR: heap 0 = VRAM, heap 1 = host RAM, heap 2 = BAR.
    fn discrete_memory_properties(bar_budget: u64, bar_usage: u64) -> MemoryProperties {
        properties(
            &[
                (1, vk::MemoryPropertyFlags::empty()),
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 0),
                (15 * 1024 * MIB, 0),
                (bar_budget, bar_usage),
            ],
        )
    }

    #[test]
    fn cpu_upload_falls_back_to_host_when_bar_exceeds_budget() {
        // 214 MiB BAR snapshot from #9745 (NVIDIA reports BAR `heapBudget = heapSize - heapUsage`,
        // so `usage + size < budget * threshold / 100` trips near half capacity).
        let properties = discrete_memory_properties(117_518_336, 106_876_928);
        let expected_host_coherent = matching_types_mask(&properties, |_, ty| {
            ty.heap_index == 1 && ty.property_flags.contains(HOST)
        });
        assert_eq!(
            properties.budgeted_memory_type_bits(
                MemoryLocation::CpuToGpu,
                all_types_mask(&properties),
                16 * MIB,
                99,
            ),
            expected_host_coherent
        );
    }

    #[test]
    fn cpu_upload_masks_out_over_budget_host_fallback() {
        let properties = properties(
            &[
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 0),
                (1000 * MIB, 995 * MIB),
                (214 * MIB, 0),
            ],
        );
        let all = all_types_mask(&properties);
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::CpuToGpu, all, 64 * MIB, 99),
            bit(3)
        );
    }

    #[test]
    fn under_budget_choices_preserve_expected_masks() {
        let properties = discrete_memory_properties(214 * MIB, 64 * 1024);
        let all = all_types_mask(&properties);
        for location in [
            MemoryLocation::GpuOnly,
            MemoryLocation::GpuToCpu,
            MemoryLocation::Unknown,
        ] {
            assert_eq!(
                properties.budgeted_memory_type_bits(location, all, 128 * MIB, 99),
                all,
                "{location:?}"
            );
        }
        let expected_cpu_to_gpu =
            matching_types_mask(&properties, |_, ty| ty.property_flags.contains(HOST));
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::CpuToGpu, all, 128 * MIB, 99),
            expected_cpu_to_gpu
        );
    }

    #[test]
    fn compatibility_mask_and_required_flags() {
        let properties = properties(
            &[
                (1, vk::MemoryPropertyFlags::empty()),
                (1, NON_COHERENT_HOST),
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 0),
                (15 * 1024 * MIB, 0),
                (117_518_336, 106_876_928),
            ],
        );
        for location in [
            MemoryLocation::GpuOnly,
            MemoryLocation::CpuToGpu,
            MemoryLocation::GpuToCpu,
        ] {
            assert_eq!(
                properties.budgeted_memory_type_bits(location, 0, 16, 99),
                0,
                "{location:?}"
            );
            assert_eq!(
                properties.budgeted_memory_type_bits(location, bit(0) | bit(1), 16, 99),
                0,
                "{location:?}"
            );
        }
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuOnly, bit(3) | bit(4), 16, 99),
            0
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(
                MemoryLocation::CpuToGpu,
                bit(0) | bit(1) | bit(2) | bit(5),
                16 * MIB,
                99,
            ),
            0
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(
                MemoryLocation::CpuToGpu,
                bit(1) | bit(4),
                128 * MIB,
                99
            ),
            bit(4)
        );
        // `Unknown` requires no flags and is all-or-nothing on the first allowed type's heap
        // (`bit(0)` healthy keeps exhausted `bit(5)` in the returned mask).
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::Unknown, 0, 16, 99),
            0
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(
                MemoryLocation::Unknown,
                bit(0) | bit(5),
                16 * MIB,
                99,
            ),
            bit(0) | bit(5)
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::Unknown, bit(5), 16 * MIB, 99),
            0
        );
    }

    #[test]
    fn gpu_only_does_not_spill_into_bar_or_host() {
        // Healthy `empty()`/`HOST` before exhausted `DEVICE_LOCAL` catches `GpuOnly` matching
        // without `DEVICE_LOCAL`, and healthy `LOCAL_HOST` after it catches spilling onto BAR.
        let properties = properties(
            &[
                (1, vk::MemoryPropertyFlags::empty()),
                (1, HOST),
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 10 * 1024 * MIB),
                (15 * 1024 * MIB, 0),
                (256 * MIB, 0),
            ],
        );
        let all = all_types_mask(&properties);
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuOnly, all, 64 * MIB, 99),
            0
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(
                MemoryLocation::Unknown,
                bit(2) | bit(3),
                64 * MIB,
                99,
            ),
            0
        );
    }

    #[test]
    fn readback_checks_cached_before_uncached_fallback() {
        // Place healthy uncached `HOST` before exhausted `CACHED_HOST` so dropping `HOST_CACHED`
        // would pick heap 1.
        let properties = properties(
            &[(1, HOST), (0, CACHED_HOST), (2, LOCAL_HOST)],
            &[(100, 90), (100, 0), (100, 0)],
        );
        let all = all_types_mask(&properties);
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuToCpu, all, 20, 100),
            0
        );

        let uncached_mask = bit(0) | bit(2);
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuToCpu, uncached_mask, 20, 100),
            uncached_mask
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuToCpu, uncached_mask, 100, 100),
            0
        );
    }

    #[test]
    fn healthy_but_incompatible_heap_does_not_make_allocation_eligible() {
        let properties = properties(
            &[(0, vk::MemoryPropertyFlags::DEVICE_LOCAL), (1, HOST)],
            &[(100, 90), (100, 0)],
        );
        let all = all_types_mask(&properties);
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuOnly, all, 20, 100),
            0
        );
    }

    #[test]
    fn zero_budget_falls_back_to_heap_size() {
        // Intentional spec-invalid inputs: `heapIndex` 9 and zero budget/size.
        let properties = properties_with_sizes(
            &[
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (2, LOCAL_HOST),
                (9, LOCAL_HOST),
            ],
            &[
                (8 * 1024 * MIB, 0, 64 * MIB),
                (16 * 1024 * MIB, 15 * 1024 * MIB, 0),
                (214 * MIB, 0, 214 * MIB),
                (0, 0, 0),
            ],
        );
        assert!(properties.heap_is_within_budget(0, 64 * MIB, 99));
        assert!(!properties.heap_is_within_budget(0, 8 * 1024 * MIB, 99));
        assert!(!properties.heap_is_within_budget(0, 0, 0));
        assert!(!properties.heap_is_within_budget(2, 0, 99));
        assert!(!properties.heap_is_within_budget(2, 1, 99));
        assert!(!properties.heap_is_within_budget(3, 0, 99));
        assert!(!properties.heap_is_within_budget(4, 1, 99));
        assert!(!properties.heap_is_within_budget(usize::MAX, 1, 99));
        assert_eq!(
            properties.budgeted_memory_type_bits(MemoryLocation::GpuOnly, bit(0), 64 * MIB, 99),
            bit(0)
        );
        assert_eq!(
            properties.budgeted_memory_type_bits(
                MemoryLocation::CpuToGpu,
                bit(1) | bit(2) | bit(3),
                64 * MIB,
                99
            ),
            bit(1)
        );
    }

    #[test]
    fn threshold_boundary_and_percentage_math() {
        let properties = properties(&[], &[(100, 90), (199, 0), (50, 0)]);
        assert!(!properties.heap_is_within_budget(0, 0, 0));
        assert!(properties.heap_is_within_budget(0, 8, 99));
        assert!(!properties.heap_is_within_budget(0, 9, 99));
        assert!(properties.heap_is_within_budget(0, 9, 100));
        assert!(!properties.heap_is_within_budget(0, 10, 100));
        // `199 * 50 / 100 == 99` without `199 / 100` integer truncation.
        assert!(properties.heap_is_within_budget(1, 98, 50));
        assert!(!properties.heap_is_within_budget(1, 99, 50));
        assert!(properties.heap_is_within_budget(2, 24, 50));
        assert!(!properties.heap_is_within_budget(2, 25, 50));
    }

    #[test]
    fn projected_usage_and_large_percentages_do_not_overflow() {
        let properties = properties(&[], &[(u64::MAX, u64::MAX)]);
        assert!(!properties.heap_is_within_budget(0, 1, 100));
        assert!(!properties.heap_is_within_budget(0, u64::MAX, 200));
        assert!(properties.heap_is_within_budget(0, u64::MAX, 255));
    }

    #[test]
    fn device_loss_checks_allocation_classes() {
        // Full zero-budget BAR (`CpuToGpu` still fits in host) and unused zero-sized heap.
        let healthy = properties_with_sizes(
            &[
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 10 * 1024 * MIB, 2 * 1024 * MIB),
                (16 * 1024 * MIB, 15 * 1024 * MIB, 3 * 1024 * MIB),
                (214 * MIB, 0, 214 * MIB),
                (0, 0, 0),
            ],
        );
        assert_eq!(
            healthy.exhausted_device_loss_location(all_types_mask(&healthy), 99),
            None
        );

        // Tile-based UMA without a plain `DEVICE_LOCAL` type, so the lazy type may come first.
        let lazy_first = properties(
            &[(1, LAZY_LOCAL), (0, LOCAL_HOST), (0, LOCAL_CACHED_HOST)],
            &[(1000 * MIB, 0), (32 * MIB, 32 * MIB)],
        );
        assert_eq!(
            lazy_first.exhausted_device_loss_location(all_types_mask(&lazy_first), 99),
            None
        );

        // Synthetic (conforming devices expose a non-lazy `DEVICE_LOCAL` type):
        // missing `GpuOnly` is skipped.
        let host_only = properties(
            &[(1, LAZY_LOCAL), (0, HOST), (0, CACHED_HOST)],
            &[(1000, 0), (100, 100)],
        );
        assert_eq!(
            host_only.exhausted_device_loss_location(all_types_mask(&host_only), 99),
            None
        );

        let vram_with_disabled_type = properties(
            &[
                (2, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
            ],
            &[
                (1000 * MIB, 990 * MIB),
                (15 * 1024 * MIB, 0),
                (10 * 1024 * MIB, 0),
            ],
        );
        let enabled_without_type_0 = all_types_mask(&vram_with_disabled_type) & !bit(0);
        assert_eq!(
            vram_with_disabled_type.exhausted_device_loss_location(enabled_without_type_0, 99),
            Some(MemoryLocation::GpuOnly)
        );

        let host_heap_exhausted_bar_healthy = properties(
            &[
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 0),
                (1000 * MIB, 990 * MIB),
                (256 * MIB, 0),
            ],
        );
        assert_eq!(
            host_heap_exhausted_bar_healthy.exhausted_device_loss_location(
                all_types_mask(&host_heap_exhausted_bar_healthy),
                99
            ),
            Some(MemoryLocation::GpuToCpu)
        );

        let both_host_and_bar_exhausted = properties(
            &[
                (0, vk::MemoryPropertyFlags::DEVICE_LOCAL),
                (1, HOST),
                (1, CACHED_HOST),
                (2, LOCAL_HOST),
            ],
            &[
                (10 * 1024 * MIB, 0),
                (1000 * MIB, 990 * MIB),
                (256 * MIB, 255 * MIB),
            ],
        );
        assert_eq!(
            both_host_and_bar_exhausted
                .exhausted_device_loss_location(all_types_mask(&both_host_and_bar_exhausted), 99),
            Some(MemoryLocation::CpuToGpu)
        );
    }
}

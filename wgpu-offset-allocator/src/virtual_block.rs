//! The standalone [`VirtualBlock`] facade.
//!
//! Port of `VmaVirtualBlock_T` (vk_mem_alloc.h): a self-contained, ready-to-use
//! suballocator over a single block, always virtual (granularity disabled, no debug
//! margin). This is the entry point for external users who just want to carve up an
//! index space.

use crate::statistics::{DetailedStatistics, Statistics};
use crate::{
    AllocationDesc, AllocationError, AllocationHandle, AllocationInfo, CreateError, HandleError,
    Suballocator, Tlsf,
};

/// Which suballocation algorithm a [`VirtualBlock`] uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Algorithm {
    /// Two-Level Segregated Fit. The default; general-purpose and low-fragmentation.
    #[default]
    Tlsf,
}

/// A standalone virtual suballocator over a single block of `size` units.
///
/// Generic over the per-allocation user data `T`. Create with [`new`](Self::new),
/// then [`allocate`](Self::allocate) / [`free`](Self::free).
///
/// # Example
///
/// ```
/// use wgpu_offset_allocator::{AllocationDesc, VirtualBlock, Algorithm};
///
/// let mut block = VirtualBlock::<()>::new(1024, Algorithm::Tlsf).unwrap();
/// let (handle, offset) = block
///     .allocate(AllocationDesc { size: 256, alignment: 16, ..Default::default() }, ())
///     .unwrap();
/// assert_eq!(offset % 16, 0);
/// block.free(handle).unwrap();
/// assert!(block.is_empty());
/// ```
#[derive(Debug)]
pub struct VirtualBlock<T> {
    inner: Tlsf<T>,
}

impl<T> VirtualBlock<T> {
    /// Creates a virtual block of `size` units using `algorithm`.
    ///
    /// The block is always virtual: buffer-image granularity is disabled
    /// (granularity 1) and debug margins are off.
    ///
    /// # Errors
    ///
    /// Returns [`CreateError::ZeroSize`] if `size` is 0.
    pub fn new(size: u64, algorithm: Algorithm) -> Result<Self, CreateError> {
        let Algorithm::Tlsf = algorithm;
        let inner = Tlsf::new(size, 1, true, 0)?;
        Ok(VirtualBlock { inner })
    }

    /// The total size of the block.
    pub fn size(&self) -> u64 {
        self.inner.size()
    }

    /// Whether the block has no live allocations.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The number of live allocations.
    pub fn allocation_count(&self) -> usize {
        self.inner.allocation_count()
    }

    /// Allocates the allocation described by `desc`, returning the handle and its offset.
    ///
    /// See [`AllocationDesc`] for the request fields. Note that
    /// [`alloc_type`](AllocationDesc::alloc_type) is irrelevant for a virtual block
    /// (granularity is disabled) but accepted for API symmetry, and
    /// [`upper_address`](AllocationDesc::upper_address) is unsupported (TLSF returns
    /// [`AllocationError::UpperAddressUnsupported`]). `user_data` is the value to
    /// associate with the allocation.
    ///
    /// A [`desc.alignment`](AllocationDesc::alignment) of `0` is coerced to `1` (see
    /// below).
    ///
    /// # Errors
    ///
    /// See [`AllocationError`]. In particular, [`AllocationError::OutOfSpace`] is the
    /// ordinary "block full" outcome.
    pub fn allocate(
        &mut self,
        desc: AllocationDesc,
        user_data: T,
    ) -> Result<(AllocationHandle, u64), AllocationError> {
        // VMA coerces a zero alignment to 1 at the `VmaVirtualBlock` facade
        // (vk_mem_alloc.h `vmaVirtualAllocate`), treating "no alignment requirement"
        // and "byte alignment" alike. Match that here. The stricter
        // `Suballocator` trait still rejects a zero alignment with
        // `InvalidAlignment`; this coercion is deliberately confined to the facade.
        let desc = AllocationDesc {
            alignment: if desc.alignment == 0 {
                1
            } else {
                desc.alignment
            },
            ..desc
        };
        let request = self.inner.create_allocation_request(desc)?;
        let offset = request.offset;
        let handle = self.inner.alloc(request, user_data);
        Ok((handle, offset))
    }

    /// Frees the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed (double free), or
    /// from another block. On error nothing is freed. See [`HandleError`] for
    /// detection guarantees.
    pub fn free(&mut self, handle: AllocationHandle) -> Result<(), HandleError> {
        self.inner.free(handle)
    }

    /// Frees all allocations.
    pub fn clear(&mut self) {
        self.inner.clear()
    }

    /// The offset of the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed, or from another
    /// block.
    pub fn allocation_offset(&self, handle: AllocationHandle) -> Result<u64, HandleError> {
        self.inner.allocation_offset(handle)
    }

    /// Full information about the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed, or from another
    /// block.
    pub fn allocation_info(
        &self,
        handle: AllocationHandle,
    ) -> Result<AllocationInfo<T>, HandleError>
    where
        T: Clone,
    {
        self.inner.allocation_info(handle)
    }

    /// Replaces the user data of the allocation identified by `handle`.
    ///
    /// # Errors
    ///
    /// Returns [`HandleError`] if `handle` is stale, already freed, or from another
    /// block. On error the block is left unchanged.
    pub fn set_user_data(
        &mut self,
        handle: AllocationHandle,
        user_data: T,
    ) -> Result<(), HandleError> {
        self.inner.set_user_data(handle, user_data)
    }

    /// Basic statistics for this block.
    pub fn statistics(&self) -> Statistics {
        let mut stats = Statistics::default();
        self.inner.add_statistics(&mut stats);
        stats
    }

    /// Detailed statistics for this block.
    pub fn detailed_statistics(&self) -> DetailedStatistics {
        let mut stats = DetailedStatistics::default();
        self.inner.add_detailed_statistics(&mut stats);
        stats
    }

    /// Validates every internal invariant (see [`Suballocator::validate`]). Mostly
    /// useful for testing.
    ///
    /// # Errors
    ///
    /// Returns a short description of the first violated invariant, if any.
    pub fn validate(&self) -> Result<(), &'static str> {
        self.inner.validate()
    }
}

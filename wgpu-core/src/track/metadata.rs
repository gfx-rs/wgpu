//! The `ResourceMetadata` type.

use crate::{hal_api::HalApi, id::TypedId, resource::Resource, Epoch};
use bit_vec::BitVec;
use std::{borrow::Cow, marker::PhantomData, mem, sync::Arc};
use wgt::strict_assert;

/// A set of resources, holding a `Arc<T>` and epoch for each member.
///
/// Testing for membership is fast, and iterating over members is
/// reasonably fast in practice. Storage consumption is proportional
/// to the largest id index of any member, not to the number of
/// members, but a bit vector tracks occupancy, so iteration touches
/// only occupied elements.
#[derive(Debug)]
pub(super) struct ResourceMetadata<A: HalApi, I: TypedId, T: Resource<I>> {
    /// If the resource with index `i` is a member, `owned[i]` is `true`.
    owned: BitVec<usize>,

    /// A vector holding clones of members' `T`s.
    resources: Vec<Option<Arc<T>>>,

    /// This tells Rust that this type should be covariant with `A`.
    _phantom: PhantomData<(A, I)>,
}

impl<A: HalApi, I: TypedId, T: Resource<I>> ResourceMetadata<A, I, T> {
    pub(super) fn new() -> Self {
        Self {
            owned: BitVec::default(),
            resources: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Returns the number of indices we can accommodate.
    pub(super) fn size(&self) -> usize {
        self.owned.len()
    }

    pub(super) fn set_size(&mut self, size: usize) {
        self.resources.resize(size, None);
        resize_bitvec(&mut self.owned, size);
    }

    /// Ensures a given index is in bounds for all arrays and does
    /// sanity checks of the presence of a refcount.
    ///
    /// In release mode this function is completely empty and is removed.
    #[cfg_attr(not(feature = "strict_asserts"), allow(unused_variables))]
    pub(super) fn tracker_assert_in_bounds(&self, index: usize) {
        strict_assert!(index < self.owned.len());
        strict_assert!(index < self.resources.len());
        strict_assert!(if self.contains(index) {
            self.resources[index].is_some()
        } else {
            true
        });
    }

    /// Returns true if the tracker owns no resources.
    ///
    /// This is a O(n) operation.
    pub(super) fn is_empty(&self) -> bool {
        !self.owned.any()
    }

    /// Returns true if the set contains the resource with the given index.
    pub(super) fn contains(&self, index: usize) -> bool {
        self.owned[index]
    }

    /// Returns true if the set contains the resource with the given index.
    ///
    /// # Safety
    ///
    /// The given `index` must be in bounds for this `ResourceMetadata`'s
    /// existing tables. See `tracker_assert_in_bounds`.
    #[inline(always)]
    pub(super) unsafe fn contains_unchecked(&self, index: usize) -> bool {
        unsafe { self.owned.get(index).unwrap_unchecked() }
    }

    /// Insert a resource into the set.
    ///
    /// Add the resource with the given index, epoch, and reference count to the
    /// set.
    ///
    /// # Safety
    ///
    /// The given `index` must be in bounds for this `ResourceMetadata`'s
    /// existing tables. See `tracker_assert_in_bounds`.
    #[inline(always)]
    pub(super) unsafe fn insert(&mut self, index: usize, resource: Arc<T>) {
        self.owned.set(index, true);
        unsafe {
            *self.resources.get_unchecked_mut(index) = Some(resource);
        }
    }

    /// Get the resource with the given index.
    ///
    /// # Safety
    ///
    /// The given `index` must be in bounds for this `ResourceMetadata`'s
    /// existing tables. See `tracker_assert_in_bounds`.
    #[inline(always)]
    pub(super) unsafe fn get_resource_unchecked(&self, index: usize) -> &Arc<T> {
        unsafe {
            self.resources
                .get_unchecked(index)
                .as_ref()
                .unwrap_unchecked()
        }
    }

    /// Get the reference count of the resource with the given index.
    ///
    /// # Safety
    ///
    /// The given `index` must be in bounds for this `ResourceMetadata`'s
    /// existing tables. See `tracker_assert_in_bounds`.
    #[inline(always)]
    pub(super) unsafe fn get_ref_count_unchecked(&self, index: usize) -> usize {
        unsafe { Arc::strong_count(self.get_resource_unchecked(index)) }
    }

    /// Returns an iterator over the resources owned by `self`.
    pub(super) fn owned_resources(&self) -> impl Iterator<Item = Arc<T>> + '_ {
        if !self.owned.is_empty() {
            self.tracker_assert_in_bounds(self.owned.len() - 1)
        };
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let resource = unsafe { self.resources.get_unchecked(index) };
            resource.as_ref().unwrap().clone()
        })
    }

    /// Returns an iterator over the resources owned by `self`.
    pub(super) fn drain_resources(&mut self) -> Vec<Arc<T>> {
        if !self.owned.is_empty() {
            self.tracker_assert_in_bounds(self.owned.len() - 1)
        };
        let mut resources = Vec::new();
        iterate_bitvec_indices(&self.owned).for_each(|index| {
            let resource = unsafe { self.resources.get_unchecked(index) };
            resources.push(resource.as_ref().unwrap().clone());
        });
        self.owned.clear();
        self.resources.clear();
        resources
    }

    /// Returns an iterator over the indices of all resources owned by `self`.
    pub(super) fn owned_indices(&self) -> impl Iterator<Item = usize> + '_ {
        if !self.owned.is_empty() {
            self.tracker_assert_in_bounds(self.owned.len() - 1)
        };
        iterate_bitvec_indices(&self.owned)
    }

    /// Remove the resource with the given index from the set.
    pub(super) unsafe fn remove(&mut self, index: usize) {
        unsafe {
            *self.resources.get_unchecked_mut(index) = None;
        }
        self.owned.set(index, false);
    }
}

/// A source of resource metadata.
///
/// This is used to abstract over the various places
/// trackers can get new resource metadata from.
pub(super) enum ResourceMetadataProvider<'a, A: HalApi, I: TypedId, T: Resource<I>> {
    /// Comes directly from explicit values.
    Direct { resource: Cow<'a, Arc<T>> },
    /// Comes from another metadata tracker.
    Indirect {
        metadata: &'a ResourceMetadata<A, I, T>,
    },
}
impl<A: HalApi, I: TypedId, T: Resource<I>> ResourceMetadataProvider<'_, A, I, T> {
    /// Get the epoch and an owned refcount from this.
    ///
    /// # Safety
    ///
    /// - The index must be in bounds of the metadata tracker if this uses an indirect source.
    /// - info must be Some if this uses a Resource source.
    #[inline(always)]
    pub(super) unsafe fn get_own(self, index: usize) -> Arc<T> {
        match self {
            ResourceMetadataProvider::Direct { resource } => resource.into_owned(),
            ResourceMetadataProvider::Indirect { metadata } => {
                metadata.tracker_assert_in_bounds(index);
                {
                    let resource = unsafe { metadata.resources.get_unchecked(index) };
                    unsafe { resource.clone().unwrap_unchecked() }
                }
            }
        }
    }
    /// Get the epoch from this.
    ///
    /// # Safety
    ///
    /// - The index must be in bounds of the metadata tracker if this uses an indirect source.
    #[inline(always)]
    pub(super) unsafe fn get_epoch(self, index: usize) -> Epoch {
        unsafe { self.get_own(index).as_info().id().unzip().1 }
    }
}

/// Resizes the given bitvec to the given size. I'm not sure why this is hard to do but it is.
fn resize_bitvec<B: bit_vec::BitBlock>(vec: &mut BitVec<B>, size: usize) {
    let owned_size_to_grow = size.checked_sub(vec.len());
    if let Some(delta) = owned_size_to_grow {
        if delta != 0 {
            vec.grow(delta, false);
        }
    } else {
        vec.truncate(size);
    }
}

/// Produces an iterator that yields the indexes of all bits that are set in the bitvec.
///
/// Will skip entire usize's worth of bits if they are all false.
fn iterate_bitvec_indices(ownership: &BitVec<usize>) -> impl Iterator<Item = usize> + '_ {
    const BITS_PER_BLOCK: usize = mem::size_of::<usize>() * 8;

    let size = ownership.len();

    ownership
        .blocks()
        .enumerate()
        .filter(|&(_, word)| word != 0)
        .flat_map(move |(word_index, mut word)| {
            let bit_start = word_index * BITS_PER_BLOCK;
            let bit_end = (bit_start + BITS_PER_BLOCK).min(size);

            (bit_start..bit_end).filter(move |_| {
                let active = word & 0b1 != 0;
                word >>= 1;

                active
            })
        })
}

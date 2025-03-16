//! The `ResourceMetadata` type.

use alloc::{sync::Arc, sync::Weak, vec::Vec};

use bit_vec::BitVec;
use wgt::strict_assert;

#[cold]
fn cold() {}

/// A trait for abstracting over Arc and Weak metadata when inserting
/// into a `Option`.
///
/// We want to avoid cloning or downgrading the value we're inserting
/// if it is already present in the `Option`, as refcounting is the
/// hostest bit of code in the tracker. This allows us to to insert
/// only if the value is different or not present.
///
/// Because [`ResourceMetadata`] is generic over [`Arc`] and [`Weak`],
/// we need a trait to give us the implementations for each.
pub trait Insertable: Sized {
    /// The value owned within the `Option`.
    ///
    /// This value is always `Arc<T>` where Self is `Arc<T>` or `Weak<T>`.
    type OwnedValue: Clone;

    /// Insert into the given option from an [`Arc`].
    fn insert_from_owned<'a>(option: &'a mut Option<Self>, value: &Self::OwnedValue) -> &'a Self;
    /// Insert into the given option from Self. This is only different for [`Weak`].
    fn insert_from_self<'a>(option: &'a mut Option<Self>, value: &Self) -> &'a Self;
    /// Get the inner value from whatever is in the option.
    ///
    /// This always clones the value, but it is only called in the error case.
    fn get_inner(option: &Self) -> Self::OwnedValue;
}

impl<T> Insertable for Arc<T> {
    type OwnedValue = Arc<T>;

    #[inline(always)]
    fn insert_from_owned<'a>(option: &'a mut Option<Arc<T>>, new_value: &Arc<T>) -> &'a Arc<T> {
        match option {
            Some(old_value) => {
                if !Arc::ptr_eq(old_value, new_value) {
                    // This should never happen. Because owning trackers keep their
                    // values alive, the value at a given tracker index should never
                    // change. We build in this check though to prevent bugs from
                    // causing silent corruption. However we mark it cold.
                    cold();

                    // Only clone and insert if the value is different.
                    *old_value = new_value.clone();
                }
                old_value
            }
            None => {
                // If the option is empty, we must clone and insert the new value.
                option.insert(new_value.clone())
            }
        }
    }

    #[inline(always)]
    fn insert_from_self<'a>(option: &'a mut Option<Arc<T>>, new_value: &Arc<T>) -> &'a Arc<T> {
        // `insert_from_owned` is the same as `insert_from_self` for Arc
        Self::insert_from_owned(option, new_value)
    }

    #[inline(always)]
    fn get_inner<'a>(option: &'a Arc<T>) -> Arc<T> {
        option.clone()
    }
}

impl<T> Insertable for Weak<T> {
    type OwnedValue = Arc<T>;

    #[inline(always)]
    fn insert_from_owned<'a>(option: &'a mut Option<Weak<T>>, new_value: &Arc<T>) -> &'a Weak<T> {
        match option {
            Some(old_value) => {
                if old_value.as_ptr() != Arc::as_ptr(new_value) {
                    // Unlike strongly owned trackers, the weak trackers (device tracker)
                    // never actually gets its items removed. We need to be able to overwrite
                    // the value in the tracker.

                    // Only downgrade and insert if the value is different.
                    *old_value = Arc::downgrade(new_value);
                }
                old_value
            }
            None => {
                // If the option is empty, we must downgrade and insert the new value.
                option.insert(Arc::downgrade(new_value))
            }
        }
    }

    #[inline(always)]
    fn insert_from_self<'a>(option: &'a mut Option<Weak<T>>, new_value: &Weak<T>) -> &'a Weak<T> {
        match option {
            Some(old_value) => {
                if old_value.as_ptr() != new_value.as_ptr() {
                    // Only clone and insert if the value is different.
                    *old_value = new_value.clone();
                }
                old_value
            }
            None => {
                // If the option is empty, we must clone and insert the new value.
                option.insert(new_value.clone())
            }
        }
    }

    #[inline(always)]
    fn get_inner<'a>(option: &'a Weak<T>) -> Arc<T> {
        option.upgrade().unwrap()
    }
}

/// A set of resources, holding a `Arc<T>` and epoch for each member.
///
/// Testing for membership is fast, and iterating over members is
/// reasonably fast in practice. Storage consumption is proportional
/// to the largest id index of any member, not to the number of
/// members, but a bit vector tracks occupancy, so iteration touches
/// only occupied elements.
#[derive(Debug)]
pub(super) struct ResourceMetadata<T> {
    /// If the resource with index `i` is a member, `owned[i]` is `true`.
    owned: BitVec<usize>,

    /// A vector holding clones of members' `T`s.
    resources: Vec<Option<T>>,
}

impl<T: Insertable> ResourceMetadata<T> {
    pub(super) fn new() -> Self {
        Self {
            owned: BitVec::default(),
            resources: Vec::new(),
        }
    }

    pub(super) fn set_size(&mut self, size: usize) {
        self.resources.resize_with(size, || None);
        resize_bitvec(&mut self.owned, size);
    }

    pub(super) fn clear(&mut self) {
        self.resources.clear();
        self.owned.clear();
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
        self.owned.get(index).unwrap_or(false)
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
    /// Returns a reference to the newly inserted resource.
    /// (This allows avoiding a clone/reference count increase in many cases.)
    ///
    /// # Safety
    ///
    /// The given `index` must be in bounds for this `ResourceMetadata`'s
    /// existing tables. See `tracker_assert_in_bounds`.
    #[inline(always)]
    pub(super) unsafe fn insert(
        &mut self,
        index: usize,
        resource: ResourceMetadataProvider<'_, T>,
    ) -> &T {
        self.owned.set(index, true);

        let resource_dst = unsafe { self.resources.get_unchecked_mut(index) };

        match resource {
            ResourceMetadataProvider::Direct { resource } => {
                T::insert_from_owned(resource_dst, resource)
            }
            ResourceMetadataProvider::Indirect { metadata } => {
                let resource = unsafe { metadata.get_resource_unchecked(index) };
                T::insert_from_self(resource_dst, resource)
            }
        }
    }

    /// Get the resource with the given index.
    ///
    /// # Safety
    ///
    /// The given `index` must be in bounds for this `ResourceMetadata`'s
    /// existing tables. See `tracker_assert_in_bounds`.
    #[inline(always)]
    pub(super) unsafe fn get_resource_unchecked(&self, index: usize) -> &T {
        unsafe {
            self.resources
                .get_unchecked(index)
                .as_ref()
                .unwrap_unchecked()
        }
    }

    /// Returns an iterator over the resources owned by `self`.
    pub(super) fn owned_resources(&self) -> impl Iterator<Item = &T> + '_ {
        if !self.owned.is_empty() {
            self.tracker_assert_in_bounds(self.owned.len() - 1)
        };
        iterate_bitvec_indices(&self.owned).map(move |index| {
            let resource = unsafe { self.resources.get_unchecked(index) };
            resource.as_ref().unwrap()
        })
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
pub(super) enum ResourceMetadataProvider<'a, T: Insertable> {
    /// Comes directly from explicit values.
    Direct { resource: &'a T::OwnedValue },
    /// Comes from another metadata tracker.
    Indirect { metadata: &'a ResourceMetadata<T> },
}
impl<T> ResourceMetadataProvider<'_, T>
where
    T: Insertable,
{
    /// Get a reference to the resource from this. This clones the resource in all pathways
    /// and should not be called unless required. Currently this is only used in error paths
    /// which do not need to be fast.
    ///
    /// # Safety
    ///
    /// - The index must be in bounds of the metadata tracker if this uses an indirect source.
    #[inline(always)]
    pub(super) unsafe fn get(&self, index: usize) -> T::OwnedValue {
        match *self {
            ResourceMetadataProvider::Direct { resource } => resource.clone(),
            ResourceMetadataProvider::Indirect { metadata } => unsafe {
                T::get_inner(metadata.get_resource_unchecked(index))
            },
        }
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
    const BITS_PER_BLOCK: usize = usize::BITS as usize;

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

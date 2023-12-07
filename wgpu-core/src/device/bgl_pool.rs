use std::{
    hash::{Hash, Hasher},
    sync::{Arc, Weak},
};

use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::{
    binding_model::{self, BindGroupLayout},
    hal_api::HalApi,
    FastHashMap, FastIndexMap,
};

/// A HashMap-like structure that stores a BindGroupLayouts [`wgt::BindGroupLayoutEntry`]s.
///
/// It is hashable, so bind group layouts can be deduplicated.
#[derive(Debug, PartialEq, Eq)]
pub struct BindGroupLayoutEntryMap {
    /// We use a IndexMap here so that we can sort the entries by their binding index,
    /// guarenteeing that the hash of equivilant layouts will be the same.
    ///
    /// Once this type is created by [`BindGroupLayoutEntryMap::from_entries`], this map
    /// will in sorted order by binding index. This allows the Hash implementation to be stable.
    inner: FastIndexMap<u32, wgt::BindGroupLayoutEntry>,
}

impl Hash for BindGroupLayoutEntryMap {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // We don't need to hash the keys, since they are just extracted from the values.
        //
        // We know this is stable and will match the behavior of PartialEq as we ensure
        // that the array is sorted.
        for entry in self.inner.values() {
            entry.hash(state);
        }
    }
}

impl BindGroupLayoutEntryMap {
    /// Create a new [`BindGroupLayoutEntryMap`] from a slice of [`wgt::BindGroupLayoutEntry`]s.
    ///
    /// Errors if there are duplicate bindings or if any binding index is greater than
    /// the device's limits.
    pub fn from_entries(
        device_limits: &wgt::Limits,
        entries: &[wgt::BindGroupLayoutEntry],
    ) -> Result<Self, binding_model::CreateBindGroupLayoutError> {
        let mut inner = FastIndexMap::with_capacity_and_hasher(entries.len(), Default::default());
        for entry in entries {
            if entry.binding > device_limits.max_bindings_per_bind_group {
                return Err(
                    binding_model::CreateBindGroupLayoutError::InvalidBindingIndex {
                        binding: entry.binding,
                        maximum: device_limits.max_bindings_per_bind_group,
                    },
                );
            }
            if inner.insert(entry.binding, *entry).is_some() {
                return Err(binding_model::CreateBindGroupLayoutError::ConflictBinding(
                    entry.binding,
                ));
            }
        }
        inner.sort_unstable_keys();

        Ok(Self { inner })
    }

    /// Get the count of [`wgt::BindGroupLayoutEntry`]s in this map.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get the [`wgt::BindGroupLayoutEntry`] for the given binding index.
    pub fn get(&self, binding: u32) -> Option<&wgt::BindGroupLayoutEntry> {
        self.inner.get(&binding)
    }

    /// Iterator over all the binding indices in this map.
    ///
    /// They will be in sorted order.
    pub fn indices(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.inner.keys().copied()
    }

    /// Iterator over all the [`wgt::BindGroupLayoutEntry`]s in this map.
    ///
    /// They will be in sorted order by binding index.
    pub fn entries(&self) -> impl ExactSizeIterator<Item = &wgt::BindGroupLayoutEntry> + '_ {
        self.inner.values()
    }
}

type SlotInner<A> = Option<Weak<BindGroupLayout<A>>>;
type BindGroupLayoutPoolSlot<A> = Arc<Mutex<SlotInner<A>>>;

pub struct BindGroupLayoutPool<A: HalApi> {
    inner: Mutex<FastHashMap<BindGroupLayoutEntryMap, BindGroupLayoutPoolSlot<A>>>,
}

impl<A: HalApi> BindGroupLayoutPool<A> {
    pub fn new() -> Self {
        Self {
            inner: FastHashMap::default(),
        }
    }

    /// Get a [`BindGroupLayout`] from the pool with the given entry map, or create a new one if it doesn't exist.
    ///
    /// Calls `f` to create a new [`BindGroupLayout`] if one doesn't exist.
    pub fn get_or_init<F, E>(
        &self,
        entry_map: &BindGroupLayoutEntryMap,
        mut f: F,
    ) -> Result<Arc<BindGroupLayout<A>>, E>
    where
        F: FnMut() -> Result<Arc<BindGroupLayout<A>>, E>,
    {
        // We have 4 potential race states
        // - There is no entry in the map
        // - There is no entry in the map, and another thread is creating one
        // - There is an entry in the map.
        // - There is an entry in the map, but it is actively being dropped.

        let mut map_guard = self.inner.lock();

        let mut entry_guard = match map_guard.get(entry_map) {
            // An entry exists for this BGL. This entry cannot be removed, however the strong refs might
            // die while we're processing.
            Some(entry) => Mutex::lock_arc(&entry),
            None => {
                let entry = Arc::new(Mutex::new(None));
                let locked = Mutex::lock_arc(&entry);
                map_guard.insert(entry_map.clone(), entry);
            }
        };

        drop(map_guard);

        let entry: &mut SlotInner<A> = &mut *entry_guard;
        
        if let Some(entry) = entry {
            // Try to upgrade the weak ref. If it succeeds, we directly return the BGL.
            if let Some(bgl) = Weak::upgrade(entry) {
                return Ok(bgl);
            }

            // The upgrade fails, the BGL is trying to drop itself on another thread. However we have the 
            // entry guard, which will block the other thread from removing its entry from the map.
        }

        
    }

    fn remove(&self, entry_map: &BindGroupLayoutEntryMap) {
        let map_guard = self.inner.lock();

        if let Some(entry) = map_guard.get(entry_map) {
            // Before we can remove the entry, we need to make sure that it isn't in the process of being re-created.
            //
            // At this point, we are in the drop implementation of the BGL, so Weak::upgrade will have previously started fail, meaning
            // another thread could be starting the re-creation process. That thread always calls upgrade inside the entry guard, so
            // once we get access to the entry guard, we know that no threads are trying to re-create the BGL.
            let entry_guard = Mutex::lock_arc(&entry);
            if entry_guard.is_none() {
                map_guard.remove(entry_map);
            }
        }
    }
}

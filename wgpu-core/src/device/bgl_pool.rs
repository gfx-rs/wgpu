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
#[derive(Debug, Default, Clone, PartialEq, Eq)]
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
    pub fn values(&self) -> impl ExactSizeIterator<Item = &wgt::BindGroupLayoutEntry> + '_ {
        self.inner.values()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = (&u32, &wgt::BindGroupLayoutEntry)> + '_ {
        self.inner.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn entry(
        &mut self,
        binding: u32,
    ) -> indexmap::map::Entry<'_, u32, wgt::BindGroupLayoutEntry> {
        self.inner.entry(binding)
    }

    pub fn contains_key(&self, id: u32) -> bool {
        self.inner.contains_key(&id)
    }
}

type SlotInner<A> = Weak<BindGroupLayout<A>>;
type BindGroupLayoutPoolSlot<A> = Arc<OnceCell<SlotInner<A>>>;

pub struct BindGroupLayoutPool<A: HalApi> {
    inner: Mutex<FastHashMap<BindGroupLayoutEntryMap, BindGroupLayoutPoolSlot<A>>>,
}

impl<A: HalApi> BindGroupLayoutPool<A> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(FastHashMap::default()),
        }
    }

    /// Get a [`BindGroupLayout`] from the pool with the given entry map, or create a new one if it doesn't exist.
    ///
    /// Behaves such that only one [`BindGroupLayout`] will be created for each unique entry map at any one time.
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
        'race: loop {
            let mut map_guard = self.inner.lock();

            let entry = match map_guard.get(entry_map) {
                // An entry exists for this BGL.
                //
                // We know that either:
                // - The BGL is still alive, and Weak::upgrade will succeed.
                // - The BGL is in the process of being dropped, and Weak::upgrade will fail.
                //
                // The entry will never be empty while the BGL is still alive.
                Some(entry) => Arc::clone(&entry),
                // No entry exists for this BGL.
                //
                // We know that the BGL is not alive, so we can create a new entry.
                None => {
                    let entry = Arc::new(OnceCell::new());
                    map_guard.insert(entry_map.clone(), Arc::clone(&entry));
                    entry
                }
            };

            drop(map_guard);

            // Some other thread may beat us to initializing the entry, but OnceCell guarentees that only one thread
            // will actually initialize the entry.
            //
            // We pass the strong reference outside of the closure to keep it alive while we're the only one keeping a reference to it.
            let mut strong = None;
            let weak = entry.get_or_try_init(|| {
                let strong_inner = f()?;
                let weak = Arc::downgrade(&strong_inner);
                strong = Some(strong_inner);
                Ok(weak)
            })?;

            // If strong is Some, that means we just initialized the entry, so we can just return it.
            if let Some(strong) = strong {
                return Ok(strong);
            }

            // The entry was already initialized by someone else, so we need to try to upgrade it.
            if let Some(strong) = weak.upgrade() {
                // We succeed, the BGL is still alive, just return that.
                return Ok(strong);
            }

            // The BGL is in the process of being dropped, because9 upgrade failed. The entry still exists in the map, but it points to nothing.
            //
            // We're in a race with the drop implementation of the BGL, so lets just go around again. When we go around again:
            // - If the entry exists, we might need to go around a few more times.
            // - If the entry doesn't exist, we'll create a new one.
            continue 'race;
        }
    }

    /// Remove the given entry map from the pool.
    ///
    /// Must *only* be called in the Drop impl of [`BindGroupLayout`].
    pub fn remove(&self, entry_map: &BindGroupLayoutEntryMap) {
        let mut map_guard = self.inner.lock();

        // Weak::upgrade will be failing long before this code is called. All threads trying to access the BGL will be spinning,
        // waiting for the entry to be removed. It is safe to remove the entry from the map.
        map_guard.remove(&entry_map);
    }
}

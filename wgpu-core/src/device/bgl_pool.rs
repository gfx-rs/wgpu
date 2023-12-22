use std::hash::{Hash, Hasher};

use crate::{
    binding_model::{self},
    FastIndexMap,
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

use std::hash::Hash;
use std::sync::{Arc, Weak};

use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::FastHashMap;

type SlotInner<V> = Weak<V>;
type ResourcePoolSlot<V> = Arc<OnceCell<SlotInner<V>>>;

pub struct ResourcePool<K, V> {
    inner: Mutex<FastHashMap<K, ResourcePoolSlot<V>>>,
}

impl<K: Clone + Eq + Hash, V> ResourcePool<K, V> {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(FastHashMap::default()),
        }
    }

    /// Get a resource from the pool with the given entry map, or create a new one if it doesn't exist using the given constructor.
    ///
    /// Behaves such that only one resource will be created for each unique entry map at any one time.
    pub fn get_or_init<F, E>(&self, key: K, constructor: F) -> Result<Arc<V>, E>
    where
        F: FnOnce(K) -> Result<Arc<V>, E>,
    {
        // We can't prove at compile time that these will only ever be consumed once,
        // so we need to do the check at runtime.
        let mut key = Some(key);
        let mut constructor = Some(constructor);

        'race: loop {
            let mut map_guard = self.inner.lock();

            let entry = match map_guard.get(key.as_ref().unwrap()) {
                // An entry exists for this resource.
                //
                // We know that either:
                // - The resource is still alive, and Weak::upgrade will succeed.
                // - The resource is in the process of being dropped, and Weak::upgrade will fail.
                //
                // The entry will never be empty while the BGL is still alive.
                Some(entry) => Arc::clone(entry),
                // No entry exists for this resource.
                //
                // We know that the resource is not alive, so we can create a new entry.
                None => {
                    let entry = Arc::new(OnceCell::new());
                    map_guard.insert(key.clone().unwrap(), Arc::clone(&entry));
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
                let strong_inner = constructor.take().unwrap()(key.take().unwrap())?;
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
                // We succeed, the resource is still alive, just return that.
                return Ok(strong);
            }

            // The resource is in the process of being dropped, because upgrade failed. The entry still exists in the map, but it points to nothing.
            //
            // We're in a race with the drop implementation of the resource, so lets just go around again. When we go around again:
            // - If the entry exists, we might need to go around a few more times.
            // - If the entry doesn't exist, we'll create a new one.
            continue 'race;
        }
    }

    /// Remove the given entry map from the pool.
    ///
    /// Must *only* be called in the Drop impl of [`BindGroupLayout`].
    pub fn remove(&self, key: &K) {
        let mut map_guard = self.inner.lock();

        // Weak::upgrade will be failing long before this code is called. All threads trying to access the resource will be spinning,
        // waiting for the entry to be removed. It is safe to remove the entry from the map.
        map_guard.remove(key);
    }
}

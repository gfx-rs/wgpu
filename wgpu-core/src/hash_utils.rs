//! Module for hashing utilities.
//!
//! Named hash_utils to prevent clashing with the std::hash module.

/// HashMap using a fast, non-cryptographic hash algorithm.
pub type FastHashMap<K, V> =
    std::collections::HashMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;
/// HashSet using a fast, non-cryptographic hash algorithm.
pub type FastHashSet<K> =
    std::collections::HashSet<K, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// IndexMap using a fast, non-cryptographic hash algorithm.
pub type FastIndexMap<K, V> =
    indexmap::IndexMap<K, V, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

/// HashMap that uses pre-hashed keys and an identity hasher.
///
/// This is useful when you only need the key to lookup the value, and don't need to store the key,
/// particularly when the key is large.
pub type PreHashedMap<K, V> =
    std::collections::HashMap<PreHashedKey<K>, V, std::hash::BuildHasherDefault<IdentityHasher>>;

/// A pre-hashed key using FxHash which allows the hashing operation to be disconnected
/// from the storage in the map.
pub struct PreHashedKey<K>(u64, std::marker::PhantomData<fn() -> K>);

impl<K> std::fmt::Debug for PreHashedKey<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("PreHashedKey").field(&self.0).finish()
    }
}

impl<K> Copy for PreHashedKey<K> {}

impl<K> Clone for PreHashedKey<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> PartialEq for PreHashedKey<K> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<K> Eq for PreHashedKey<K> {}

impl<K> std::hash::Hash for PreHashedKey<K> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

impl<K: std::hash::Hash> PreHashedKey<K> {
    pub fn from_key(key: &K) -> Self {
        use std::hash::Hasher;

        let mut hasher = rustc_hash::FxHasher::default();
        key.hash(&mut hasher);
        Self(hasher.finish(), std::marker::PhantomData)
    }
}

/// A hasher which does nothing. Useful for when you want to use a map with pre-hashed keys.
///
/// When hashing with this hasher, you must provide exactly 8 bytes. Multiple calls to `write`
/// will overwrite the previous value.
#[derive(Default)]
pub struct IdentityHasher {
    hash: u64,
}

impl std::hash::Hasher for IdentityHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.hash = u64::from_ne_bytes(
            bytes
                .try_into()
                .expect("identity hasher must be given exactly 8 bytes"),
        );
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

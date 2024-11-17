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

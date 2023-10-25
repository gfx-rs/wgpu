/*!
Reusing collections' previous allocations.
*/

/// A value that can be reset to its initial state, retaining its current allocations.
///
/// Naga attempts to lower the cost of SPIR-V generation by allowing clients to
/// reuse the same `Writer` for multiple Module translations. Reusing a `Writer`
/// means that the `Vec`s, `HashMap`s, and other heap-allocated structures the
/// `Writer` uses internally begin the translation with heap-allocated buffers
/// ready to use.
///
/// But this approach introduces the risk of `Writer` state leaking from one
/// module to the next. When a developer adds fields to `Writer` or its internal
/// types, they must remember to reset their contents between modules.
///
/// One trick to ensure that every field has been accounted for is to use Rust's
/// struct literal syntax to construct a new, reset value. If a developer adds a
/// field, but neglects to update the reset code, the compiler will complain
/// that a field is missing from the literal. This trait's `recycle` method
/// takes `self` by value, and returns `Self` by value, encouraging the use of
/// struct literal expressions in its implementation.
pub trait Recyclable {
    /// Clear `self`, retaining its current memory allocations.
    ///
    /// Shrink the buffer if it's currently much larger than was actually used.
    /// This prevents a module with exceptionally large allocations from causing
    /// the `Writer` to retain more memory than it needs indefinitely.
    fn recycle(self) -> Self;
}

// Stock values for various collections.

impl<T> Recyclable for Vec<T> {
    fn recycle(mut self) -> Self {
        self.clear();
        self
    }
}

impl<K, V, S: Clone> Recyclable for std::collections::HashMap<K, V, S> {
    fn recycle(mut self) -> Self {
        self.clear();
        self
    }
}

impl<K, S: Clone> Recyclable for std::collections::HashSet<K, S> {
    fn recycle(mut self) -> Self {
        self.clear();
        self
    }
}

impl<K, S: Clone> Recyclable for indexmap::IndexSet<K, S> {
    fn recycle(mut self) -> Self {
        self.clear();
        self
    }
}

impl<K: Ord, V> Recyclable for std::collections::BTreeMap<K, V> {
    fn recycle(mut self) -> Self {
        self.clear();
        self
    }
}

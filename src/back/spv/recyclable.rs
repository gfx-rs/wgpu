//! Reusing collections' previous allocations.

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

/// Maximum extra capacity that a recycled vector is allowed to have. If the
/// actual capacity is larger, we re-allocate the vector storage with lower
/// capacity.
const MAX_EXTRA_CAPACITY_PERCENT: usize = 200;

/// Minimum extra capacity to keep when re-allocating the vector storage.
const MIN_EXTRA_CAPACITY_PERCENT: usize = 20;

/// Minimum sensible length to consider for re-allocation.
const MIN_LENGTH: usize = 16;

impl<T> Recyclable for Vec<T> {
    fn recycle(mut self) -> Self {
        let extra_capacity = (self.capacity() - self.len()) * 100 / self.len().max(MIN_LENGTH);

        if extra_capacity > MAX_EXTRA_CAPACITY_PERCENT {
            //TODO: use `shrink_to` when it's stable
            self = Vec::with_capacity(self.len() + self.len() * MIN_EXTRA_CAPACITY_PERCENT / 100);
        } else {
            self.clear();
        }

        self
    }
}

impl<K, V, S: Clone> Recyclable for std::collections::HashMap<K, V, S> {
    fn recycle(mut self) -> Self {
        let extra_capacity = (self.capacity() - self.len()) * 100 / self.len().max(MIN_LENGTH);

        if extra_capacity > MAX_EXTRA_CAPACITY_PERCENT {
            //TODO: use `shrink_to` when it's stable
            self = Self::with_capacity_and_hasher(
                self.len() + self.len() * MIN_EXTRA_CAPACITY_PERCENT / 100,
                self.hasher().clone(),
            );
        } else {
            self.clear();
        }

        self
    }
}

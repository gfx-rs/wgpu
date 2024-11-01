//! Module containing the [`WeakVec`] API.

use std::sync::Weak;

/// An optimized container for `Weak` references of `T` that minimizes reallocations by
/// dropping older elements that no longer have strong references to them.
#[derive(Debug)]
pub(crate) struct WeakVec<T> {
    inner: Vec<Option<Weak<T>>>,
    empty_slots: Vec<usize>,
    scan_slots_on_next_push: bool,
}

impl<T> Default for WeakVec<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
            empty_slots: Default::default(),
            scan_slots_on_next_push: false,
        }
    }
}

impl<T> WeakVec<T> {
    pub(crate) fn new() -> Self {
        Self {
            inner: Vec::new(),
            empty_slots: Vec::default(),
            scan_slots_on_next_push: false,
        }
    }

    /// Pushes a new element to this collection.
    ///
    /// If the inner Vec needs to be reallocated, we will first drop older elements that
    /// no longer have strong references to them.
    pub(crate) fn push(&mut self, value: Weak<T>) {
        if self.scan_slots_on_next_push {
            for (i, value) in self.inner.iter_mut().enumerate() {
                if let Some(w) = value {
                    if w.strong_count() == 0 {
                        *value = None;
                        self.empty_slots.push(i);
                    }
                }
            }
        }
        if let Some(i) = self.empty_slots.pop() {
            self.inner[i] = Some(value);
            self.scan_slots_on_next_push = false;
        } else {
            self.inner.push(Some(value));
            self.scan_slots_on_next_push = self.inner.len() == self.inner.capacity();
        }
    }
}

pub(crate) struct WeakVecIter<T> {
    inner: std::iter::Flatten<std::vec::IntoIter<Option<Weak<T>>>>,
}

impl<T> Iterator for WeakVecIter<T> {
    type Item = Weak<T>;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

impl<T> IntoIterator for WeakVec<T> {
    type Item = Weak<T>;
    type IntoIter = WeakVecIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        WeakVecIter {
            inner: self.inner.into_iter().flatten(),
        }
    }
}

//! Module containing the [`WeakVec`] API.

use std::sync::Weak;

/// A container that holds Weak references of T.
///
/// On `push` it scans its contents for weak references with no strong references still alive and drops them.
#[derive(Debug)]
pub(crate) struct WeakVec<T> {
    inner: Vec<Option<Weak<T>>>,
}

impl<T> Default for WeakVec<T> {
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T> WeakVec<T> {
    pub(crate) fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Pushes a new element to this collection, dropping older elements that no longer have
    /// a strong reference to them.
    ///
    /// NOTE: The length and capacity of this collection do not change when old elements are
    /// dropped.
    pub(crate) fn push(&mut self, value: Weak<T>) {
        let mut to_insert = Some(value);
        for slot in &mut self.inner {
            if let Some(w) = slot {
                if w.strong_count() == 0 {
                    *slot = to_insert.take();
                }
            } else {
                *slot = to_insert.take();
            }
        }
        if let Some(to_insert) = to_insert {
            self.inner.push(Some(to_insert));
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

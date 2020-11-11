use std::{cmp::Ordering, fmt, hash, marker::PhantomData, num::NonZeroU32};

/// An unique index in the arena array that a handle points to.
/// The "non-zero" part ensures that an `Option<Handle<T>>` has
/// the same size and representation as `Handle<T>`.
type Index = NonZeroU32;

/// A strongly typed reference to a SPIR-V element.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(transparent)
)]
pub struct Handle<T> {
    index: Index,
    #[cfg_attr(any(feature = "serialize", feature = "deserialize"), serde(skip))]
    marker: PhantomData<T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Handle {
            index: self.index,
            marker: self.marker,
        }
    }
}
impl<T> Copy for Handle<T> {}
impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}
impl<T> Eq for Handle<T> {}
impl<T> PartialOrd for Handle<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.index.partial_cmp(&other.index)
    }
}
impl<T> Ord for Handle<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.index.cmp(&other.index)
    }
}
impl<T> fmt::Debug for Handle<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "Handle({})", self.index)
    }
}
impl<T> hash::Hash for Handle<T> {
    fn hash<H: hash::Hasher>(&self, hasher: &mut H) {
        self.index.hash(hasher)
    }
}

impl<T> Handle<T> {
    #[cfg(test)]
    pub const DUMMY: Self = Handle {
        index: unsafe { NonZeroU32::new_unchecked(!0) },
        marker: PhantomData,
    };

    pub(crate) fn new(index: Index) -> Self {
        Handle {
            index,
            marker: PhantomData,
        }
    }

    /// Returns the zero-based index of this handle.
    pub fn index(self) -> usize {
        let index = self.index.get() - 1;
        index as usize
    }
}

/// An arena holding some kind of component (e.g., type, constant,
/// instruction, etc.) that can be referenced.
///
/// Adding new items to the arena produces a strongly-typed [`Handle`].
/// The arena can be indexed using the given handle to obtain
/// a reference to the stored item.
#[derive(Debug)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(transparent)
)]
#[cfg_attr(test, derive(PartialEq))]
pub struct Arena<T> {
    /// Values of this arena.
    data: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    /// Create a new arena with no initial capacity allocated.
    pub fn new() -> Self {
        Arena { data: Vec::new() }
    }

    /// Returns the current number of items stored in this arena.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the arena contains no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns an iterator over the items stored in this arena, returning both
    /// the item's handle and a reference to it.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &T)> {
        self.data.iter().enumerate().map(|(i, v)| {
            let position = i + 1;
            let index = unsafe { Index::new_unchecked(position as u32) };
            (Handle::new(index), v)
        })
    }

    /// Adds a new value to the arena, returning a typed handle.
    ///
    /// The value is not linked to any SPIR-V module.
    pub fn append(&mut self, value: T) -> Handle<T> {
        let position = self.data.len() + 1;
        let index = unsafe { Index::new_unchecked(position as u32) };
        self.data.push(value);
        Handle::new(index)
    }

    /// Fetch a handle to an existing type.
    pub fn fetch_if<F: Fn(&T) -> bool>(&self, fun: F) -> Option<Handle<T>> {
        self.data
            .iter()
            .position(fun)
            .map(|index| Handle::new(unsafe { Index::new_unchecked((index + 1) as u32) }))
    }

    /// Adds a value with a custom check for uniqueness:
    /// returns a handle pointing to
    /// an existing element if the check succeeds, or adds a new
    /// element otherwise.
    pub fn fetch_if_or_append<F: Fn(&T, &T) -> bool>(&mut self, value: T, fun: F) -> Handle<T> {
        if let Some(index) = self.data.iter().position(|d| fun(d, &value)) {
            let index = unsafe { Index::new_unchecked((index + 1) as u32) };
            Handle::new(index)
        } else {
            self.append(value)
        }
    }

    /// Adds a value with a check for uniqueness, where the check is plain comparison.
    pub fn fetch_or_append(&mut self, value: T) -> Handle<T>
    where
        T: PartialEq,
    {
        self.fetch_if_or_append(value, T::eq)
    }

    pub fn try_get(&self, handle: Handle<T>) -> Option<&T> {
        self.data.get(handle.index.get() as usize - 1)
    }

    /// Get a mutable reference to an element in the arena.
    pub fn get_mut(&mut self, handle: Handle<T>) -> &mut T {
        self.data.get_mut(handle.index.get() as usize - 1).unwrap()
    }
}

impl<T> std::ops::Index<Handle<T>> for Arena<T> {
    type Output = T;
    fn index(&self, handle: Handle<T>) -> &T {
        let index = handle.index.get() - 1;
        &self.data[index as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0);
        let t2 = arena.append(0);
        assert!(t1 != t2);
        assert!(arena[t1] == arena[t2]);
    }

    #[test]
    fn append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0);
        let t2 = arena.append(1);
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
    }

    #[test]
    fn fetch_or_append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0);
        let t2 = arena.fetch_or_append(0);
        assert!(t1 == t2);
        assert!(arena[t1] == arena[t2])
    }

    #[test]
    fn fetch_or_append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0);
        let t2 = arena.fetch_or_append(1);
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
    }
}

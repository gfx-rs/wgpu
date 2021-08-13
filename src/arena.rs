use std::{cmp::Ordering, fmt, hash, marker::PhantomData, num::NonZeroU32, ops};

/// An unique index in the arena array that a handle points to.
/// The "non-zero" part ensures that an `Option<Handle<T>>` has
/// the same size and representation as `Handle<T>`.
type Index = NonZeroU32;

use crate::Span;

/// A strongly typed reference to an arena item.
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
        write!(formatter, "[{}]", self.index)
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

/// A strongly typed range of handles.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(transparent)
)]
pub struct Range<T> {
    inner: ops::Range<u32>,
    #[cfg_attr(any(feature = "serialize", feature = "deserialize"), serde(skip))]
    marker: PhantomData<T>,
}

impl<T> Clone for Range<T> {
    fn clone(&self) -> Self {
        Range {
            inner: self.inner.clone(),
            marker: self.marker,
        }
    }
}
impl<T> fmt::Debug for Range<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "[{}..{}]", self.inner.start + 1, self.inner.end)
    }
}
impl<T> Iterator for Range<T> {
    type Item = Handle<T>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.inner.start < self.inner.end {
            self.inner.start += 1;
            Some(Handle {
                index: NonZeroU32::new(self.inner.start).unwrap(),
                marker: self.marker,
            })
        } else {
            None
        }
    }
}

/// An arena holding some kind of component (e.g., type, constant,
/// instruction, etc.) that can be referenced.
///
/// Adding new items to the arena produces a strongly-typed [`Handle`].
/// The arena can be indexed using the given handle to obtain
/// a reference to the stored item.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "serialize", serde(transparent))]
#[cfg_attr(test, derive(PartialEq))]
pub struct Arena<T> {
    /// Values of this arena.
    data: Vec<T>,
    #[cfg(feature = "span")]
    #[cfg_attr(feature = "serialize", serde(skip))]
    span_info: Vec<Span>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<T: fmt::Debug> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<T> Arena<T> {
    /// Create a new arena with no initial capacity allocated.
    pub fn new() -> Self {
        Arena {
            data: Vec::new(),
            #[cfg(feature = "span")]
            span_info: Vec::new(),
        }
    }

    /// Extracts the inner vector.
    pub fn into_inner(self) -> Vec<T> {
        self.data
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
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Handle<T>, &T)> {
        self.data.iter().enumerate().map(|(i, v)| {
            let position = i + 1;
            let index = unsafe { Index::new_unchecked(position as u32) };
            (Handle::new(index), v)
        })
    }

    /// Returns a iterator over the items stored in this arena,
    /// returning both the item's handle and a mutable reference to it.
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Handle<T>, &mut T)> {
        self.data.iter_mut().enumerate().map(|(i, v)| {
            let position = i + 1;
            let index = unsafe { Index::new_unchecked(position as u32) };
            (Handle::new(index), v)
        })
    }

    /// Adds a new value to the arena, returning a typed handle.
    pub fn append(&mut self, value: T, span: Span) -> Handle<T> {
        #[cfg(not(feature = "span"))]
        let _ = span;
        let position = self.data.len() + 1;
        let index =
            Index::new(position as u32).expect("Failed to append to Arena. Handle overflows");
        self.data.push(value);
        #[cfg(feature = "span")]
        self.span_info.push(span);
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
    pub fn fetch_if_or_append<F: Fn(&T, &T) -> bool>(
        &mut self,
        value: T,
        span: Span,
        fun: F,
    ) -> Handle<T> {
        if let Some(index) = self.data.iter().position(|d| fun(d, &value)) {
            let index = unsafe { Index::new_unchecked((index + 1) as u32) };
            Handle::new(index)
        } else {
            self.append(value, span)
        }
    }

    /// Adds a value with a check for uniqueness, where the check is plain comparison.
    pub fn fetch_or_append(&mut self, value: T, span: Span) -> Handle<T>
    where
        T: PartialEq,
    {
        self.fetch_if_or_append(value, span, T::eq)
    }

    pub fn try_get(&self, handle: Handle<T>) -> Option<&T> {
        self.data.get(handle.index())
    }

    /// Get a mutable reference to an element in the arena.
    pub fn get_mut(&mut self, handle: Handle<T>) -> &mut T {
        self.data.get_mut(handle.index()).unwrap()
    }

    /// Get the range of handles from a particular number of elements to the end.
    pub fn range_from(&self, old_length: usize) -> Range<T> {
        Range {
            inner: old_length as u32..self.data.len() as u32,
            marker: PhantomData,
        }
    }

    /// Clears the arena keeping all allocations
    pub fn clear(&mut self) {
        self.data.clear()
    }

    pub fn get_span(&self, handle: Handle<T>) -> &Span {
        #[cfg(feature = "span")]
        {
            return self.span_info.get(handle.index()).unwrap_or(&Span::Unknown);
        }
        #[cfg(not(feature = "span"))]
        {
            let _ = handle;
            &Span::Unknown
        }
    }
}

#[cfg(feature = "deserialize")]
impl<'de, T> serde::Deserialize<'de> for Arena<T>
where
    T: serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let data = Vec::deserialize(deserializer)?;
        #[cfg(feature = "span")]
        let span_info = std::iter::repeat(Span::Unknown).take(data.len()).collect();

        Ok(Self {
            data,
            #[cfg(feature = "span")]
            span_info,
        })
    }
}

impl<T> ops::Index<Handle<T>> for Arena<T> {
    type Output = T;
    fn index(&self, handle: Handle<T>) -> &T {
        &self.data[handle.index()]
    }
}

impl<T> ops::IndexMut<Handle<T>> for Arena<T> {
    fn index_mut(&mut self, handle: Handle<T>) -> &mut T {
        &mut self.data[handle.index()]
    }
}

impl<T> ops::Index<Range<T>> for Arena<T> {
    type Output = [T];
    fn index(&self, range: Range<T>) -> &[T] {
        &self.data[range.inner.start as usize..range.inner.end as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0, Default::default());
        let t2 = arena.append(0, Default::default());
        assert!(t1 != t2);
        assert!(arena[t1] == arena[t2]);
    }

    #[test]
    fn append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.append(0, Default::default());
        let t2 = arena.append(1, Default::default());
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
    }

    #[test]
    fn fetch_or_append_non_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0, Default::default());
        let t2 = arena.fetch_or_append(0, Default::default());
        assert!(t1 == t2);
        assert!(arena[t1] == arena[t2])
    }

    #[test]
    fn fetch_or_append_unique() {
        let mut arena: Arena<u8> = Arena::new();
        let t1 = arena.fetch_or_append(0, Default::default());
        let t2 = arena.fetch_or_append(1, Default::default());
        assert!(t1 != t2);
        assert!(arena[t1] != arena[t2]);
    }
}

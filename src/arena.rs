use std::{cmp::Ordering, fmt, hash, marker::PhantomData, num::NonZeroU32, ops};

/// An unique index in the arena array that a handle points to.
/// The "non-zero" part ensures that an `Option<Handle<T>>` has
/// the same size and representation as `Handle<T>`.
type Index = NonZeroU32;

use crate::Span;
use indexmap::set::IndexSet;

#[derive(Clone, Copy, Debug, thiserror::Error, PartialEq)]
#[error("Handle {index} of {kind} is either not present, or inaccessible yet")]
pub struct BadHandle {
    pub kind: &'static str,
    pub index: usize,
}

impl BadHandle {
    fn new<T>(handle: Handle<T>) -> Self {
        Self {
            kind: std::any::type_name::<T>(),
            index: handle.index(),
        }
    }
}

/// A strongly typed reference to an arena item.
///
/// A `Handle` value can be used as an index into an [`Arena`] or [`UniqueArena`].
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(transparent)
)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
        index: unsafe { NonZeroU32::new_unchecked(u32::MAX) },
        marker: PhantomData,
    };

    pub(crate) const fn new(index: Index) -> Self {
        Handle {
            index,
            marker: PhantomData,
        }
    }

    /// Returns the zero-based index of this handle.
    pub const fn index(self) -> usize {
        let index = self.index.get() - 1;
        index as usize
    }

    /// Convert a `usize` index into a `Handle<T>`.
    fn from_usize(index: usize) -> Self {
        let handle_index = u32::try_from(index + 1)
            .ok()
            .and_then(Index::new)
            .expect("Failed to insert into arena. Handle overflows");
        Handle::new(handle_index)
    }

    /// Convert a `usize` index into a `Handle<T>`, without range checks.
    const unsafe fn from_usize_unchecked(index: usize) -> Self {
        Handle::new(Index::new_unchecked((index + 1) as u32))
    }
}

/// A strongly typed range of handles.
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(
    any(feature = "serialize", feature = "deserialize"),
    serde(transparent)
)]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
pub struct Range<T> {
    inner: ops::Range<u32>,
    #[cfg_attr(any(feature = "serialize", feature = "deserialize"), serde(skip))]
    marker: PhantomData<T>,
}

impl<T> Range<T> {
    pub(crate) const fn erase_type(self) -> Range<()> {
        let Self { inner, marker: _ } = self;
        Range {
            inner,
            marker: PhantomData,
        }
    }
}

// NOTE: Keep this diagnostic in sync with that of [`BadHandle`].
#[derive(Clone, Debug, thiserror::Error)]
#[error("Handle range {range:?} of {kind} is either not present, or inaccessible yet")]
pub struct BadRangeError {
    // This error is used for many `Handle` types, but there's no point in making this generic, so
    // we just flatten them all to `Handle<()>` here.
    kind: &'static str,
    range: Range<()>,
}

impl BadRangeError {
    pub fn new<T>(range: Range<T>) -> Self {
        Self {
            kind: std::any::type_name::<T>(),
            range: range.erase_type(),
        }
    }
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

impl<T> Range<T> {
    /// Return a range enclosing handles `first` through `last`, inclusive.
    pub fn new_from_bounds(first: Handle<T>, last: Handle<T>) -> Self {
        Self {
            inner: (first.index() as u32)..(last.index() as u32 + 1),
            marker: Default::default(),
        }
    }

    /// Return the first and last handles included in `self`.
    ///
    /// If `self` is an empty range, there are no handles included, so
    /// return `None`.
    pub fn first_and_last(&self) -> Option<(Handle<T>, Handle<T>)> {
        if self.inner.start < self.inner.end {
            Some((
                // `Range::new_from_bounds` expects a 1-based, start- and
                // end-inclusive range, but `self.inner` is a zero-based,
                // end-exclusive range.
                Handle::new(Index::new(self.inner.start + 1).unwrap()),
                Handle::new(Index::new(self.inner.end).unwrap()),
            ))
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
#[cfg_attr(feature = "clone", derive(Clone))]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "serialize", serde(transparent))]
#[cfg_attr(feature = "arbitrary", derive(arbitrary::Arbitrary))]
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
    pub const fn new() -> Self {
        Arena {
            data: Vec::new(),
            #[cfg(feature = "span")]
            span_info: Vec::new(),
        }
    }

    /// Extracts the inner vector.
    #[allow(clippy::missing_const_for_fn)] // ignore due to requirement of #![feature(const_precise_live_drops)]
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
        self.data
            .iter()
            .enumerate()
            .map(|(i, v)| unsafe { (Handle::from_usize_unchecked(i), v) })
    }

    /// Returns a iterator over the items stored in this arena,
    /// returning both the item's handle and a mutable reference to it.
    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = (Handle<T>, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, v)| unsafe { (Handle::from_usize_unchecked(i), v) })
    }

    /// Adds a new value to the arena, returning a typed handle.
    pub fn append(&mut self, value: T, span: Span) -> Handle<T> {
        #[cfg(not(feature = "span"))]
        let _ = span;
        let index = self.data.len();
        self.data.push(value);
        #[cfg(feature = "span")]
        self.span_info.push(span);
        Handle::from_usize(index)
    }

    /// Fetch a handle to an existing type.
    pub fn fetch_if<F: Fn(&T) -> bool>(&self, fun: F) -> Option<Handle<T>> {
        self.data
            .iter()
            .position(fun)
            .map(|index| unsafe { Handle::from_usize_unchecked(index) })
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
            unsafe { Handle::from_usize_unchecked(index) }
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

    pub fn try_get(&self, handle: Handle<T>) -> Result<&T, BadHandle> {
        self.data
            .get(handle.index())
            .ok_or_else(|| BadHandle::new(handle))
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

    pub fn get_span(&self, handle: Handle<T>) -> Span {
        #[cfg(feature = "span")]
        {
            *self
                .span_info
                .get(handle.index())
                .unwrap_or(&Span::default())
        }
        #[cfg(not(feature = "span"))]
        {
            let _ = handle;
            Span::default()
        }
    }

    /// Assert that `handle` is valid for this arena.
    pub fn check_contains_handle(&self, handle: Handle<T>) -> Result<(), BadHandle> {
        if handle.index() < self.data.len() {
            Ok(())
        } else {
            Err(BadHandle::new(handle))
        }
    }

    /// Assert that `range` is valid for this arena.
    pub fn check_contains_range(&self, range: &Range<T>) -> Result<(), BadRangeError> {
        // Since `range.inner` is a `Range<u32>`, we only need to
        // check that the start precedes the end, and that the end is
        // in range.
        if range.inner.start > range.inner.end
            || self
                .check_contains_handle(Handle::new(range.inner.end.try_into().unwrap()))
                .is_err()
        {
            Err(BadRangeError::new(range.clone()))
        } else {
            Ok(())
        }
    }

    #[cfg(feature = "compact")]
    pub(crate) fn retain_mut<P>(&mut self, mut predicate: P)
    where
        P: FnMut(Handle<T>, &mut T) -> bool,
    {
        let mut index = 0;
        self.data.retain_mut(|elt| {
            index += 1;
            let handle = Handle::new(Index::new(index).unwrap());
            predicate(handle, elt)
        })
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
        let span_info = std::iter::repeat(Span::default())
            .take(data.len())
            .collect();

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

/// An arena whose elements are guaranteed to be unique.
///
/// A `UniqueArena` holds a set of unique values of type `T`, each with an
/// associated [`Span`]. Inserting a value returns a `Handle<T>`, which can be
/// used to index the `UniqueArena` and obtain shared access to the `T` element.
/// Access via a `Handle` is an array lookup - no hash lookup is necessary.
///
/// The element type must implement `Eq` and `Hash`. Insertions of equivalent
/// elements, according to `Eq`, all return the same `Handle`.
///
/// Once inserted, elements may not be mutated.
///
/// `UniqueArena` is similar to [`Arena`]: If `Arena` is vector-like,
/// `UniqueArena` is `HashSet`-like.
#[cfg_attr(feature = "clone", derive(Clone))]
pub struct UniqueArena<T> {
    set: IndexSet<T>,

    /// Spans for the elements, indexed by handle.
    ///
    /// The length of this vector is always equal to `set.len()`. `IndexSet`
    /// promises that its elements "are indexed in a compact range, without
    /// holes in the range 0..set.len()", so we can always use the indices
    /// returned by insertion as indices into this vector.
    #[cfg(feature = "span")]
    span_info: Vec<Span>,
}

impl<T> UniqueArena<T> {
    /// Create a new arena with no initial capacity allocated.
    pub fn new() -> Self {
        UniqueArena {
            set: IndexSet::new(),
            #[cfg(feature = "span")]
            span_info: Vec::new(),
        }
    }

    /// Return the current number of items stored in this arena.
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Return `true` if the arena contains no elements.
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// Clears the arena, keeping all allocations.
    pub fn clear(&mut self) {
        self.set.clear();
        #[cfg(feature = "span")]
        self.span_info.clear();
    }

    /// Return the span associated with `handle`.
    ///
    /// If a value has been inserted multiple times, the span returned is the
    /// one provided with the first insertion.
    ///
    /// If the `span` feature is not enabled, always return `Span::default`.
    /// This can be detected with [`Span::is_defined`].
    pub fn get_span(&self, handle: Handle<T>) -> Span {
        #[cfg(feature = "span")]
        {
            *self
                .span_info
                .get(handle.index())
                .unwrap_or(&Span::default())
        }
        #[cfg(not(feature = "span"))]
        {
            let _ = handle;
            Span::default()
        }
    }

    #[cfg(feature = "compact")]
    pub(crate) fn drain_all(&mut self) -> UniqueArenaDrain<T> {
        UniqueArenaDrain {
            inner_elts: self.set.drain(..),
            #[cfg(feature = "span")]
            inner_spans: self.span_info.drain(..),
            index: Index::new(1).unwrap(),
        }
    }
}

#[cfg(feature = "compact")]
pub(crate) struct UniqueArenaDrain<'a, T> {
    inner_elts: indexmap::set::Drain<'a, T>,
    #[cfg(feature = "span")]
    inner_spans: std::vec::Drain<'a, Span>,
    index: Index,
}

#[cfg(feature = "compact")]
impl<'a, T> Iterator for UniqueArenaDrain<'a, T> {
    type Item = (Handle<T>, T, Span);

    fn next(&mut self) -> Option<Self::Item> {
        match self.inner_elts.next() {
            Some(elt) => {
                let handle = Handle::new(self.index);
                self.index = self.index.checked_add(1).unwrap();
                #[cfg(feature = "span")]
                let span = self.inner_spans.next().unwrap();
                #[cfg(not(feature = "span"))]
                let span = Span::default();
                Some((handle, elt, span))
            }
            None => None,
        }
    }
}

impl<T: Eq + hash::Hash> UniqueArena<T> {
    /// Returns an iterator over the items stored in this arena, returning both
    /// the item's handle and a reference to it.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (Handle<T>, &T)> {
        self.set.iter().enumerate().map(|(i, v)| {
            let position = i + 1;
            let index = unsafe { Index::new_unchecked(position as u32) };
            (Handle::new(index), v)
        })
    }

    /// Insert a new value into the arena.
    ///
    /// Return a [`Handle<T>`], which can be used to index this arena to get a
    /// shared reference to the element.
    ///
    /// If this arena already contains an element that is `Eq` to `value`,
    /// return a `Handle` to the existing element, and drop `value`.
    ///
    /// When the `span` feature is enabled, if `value` is inserted into the
    /// arena, associate `span` with it. An element's span can be retrieved with
    /// the [`get_span`] method.
    ///
    /// [`Handle<T>`]: Handle
    /// [`get_span`]: UniqueArena::get_span
    pub fn insert(&mut self, value: T, span: Span) -> Handle<T> {
        let (index, added) = self.set.insert_full(value);

        #[cfg(feature = "span")]
        {
            if added {
                debug_assert!(index == self.span_info.len());
                self.span_info.push(span);
            }

            debug_assert!(self.set.len() == self.span_info.len());
        }

        #[cfg(not(feature = "span"))]
        let _ = (span, added);

        Handle::from_usize(index)
    }

    /// Replace an old value with a new value.
    ///
    /// # Panics
    ///
    /// - if the old value is not in the arena
    /// - if the new value already exists in the arena
    pub fn replace(&mut self, old: Handle<T>, new: T) {
        let (index, added) = self.set.insert_full(new);
        assert!(added && index == self.set.len() - 1);

        self.set.swap_remove_index(old.index()).unwrap();
    }

    /// Return this arena's handle for `value`, if present.
    ///
    /// If this arena already contains an element equal to `value`,
    /// return its handle. Otherwise, return `None`.
    pub fn get(&self, value: &T) -> Option<Handle<T>> {
        self.set
            .get_index_of(value)
            .map(|index| unsafe { Handle::from_usize_unchecked(index) })
    }

    /// Return this arena's value at `handle`, if that is a valid handle.
    pub fn get_handle(&self, handle: Handle<T>) -> Result<&T, BadHandle> {
        self.set
            .get_index(handle.index())
            .ok_or_else(|| BadHandle::new(handle))
    }

    /// Assert that `handle` is valid for this arena.
    pub fn check_contains_handle(&self, handle: Handle<T>) -> Result<(), BadHandle> {
        if handle.index() < self.set.len() {
            Ok(())
        } else {
            Err(BadHandle::new(handle))
        }
    }
}

impl<T> Default for UniqueArena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: fmt::Debug + Eq + hash::Hash> fmt::Debug for UniqueArena<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

impl<T> ops::Index<Handle<T>> for UniqueArena<T> {
    type Output = T;
    fn index(&self, handle: Handle<T>) -> &T {
        &self.set[handle.index()]
    }
}

#[cfg(feature = "serialize")]
impl<T> serde::Serialize for UniqueArena<T>
where
    T: Eq + hash::Hash + serde::Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.set.serialize(serializer)
    }
}

#[cfg(feature = "deserialize")]
impl<'de, T> serde::Deserialize<'de> for UniqueArena<T>
where
    T: Eq + hash::Hash + serde::Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let set = IndexSet::deserialize(deserializer)?;
        #[cfg(feature = "span")]
        let span_info = std::iter::repeat(Span::default()).take(set.len()).collect();

        Ok(Self {
            set,
            #[cfg(feature = "span")]
            span_info,
        })
    }
}

//Note: largely borrowed from `HashSet` implementation
#[cfg(feature = "arbitrary")]
impl<'a, T> arbitrary::Arbitrary<'a> for UniqueArena<T>
where
    T: Eq + hash::Hash + arbitrary::Arbitrary<'a>,
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut arena = Self::default();
        for elem in u.arbitrary_iter()? {
            arena.set.insert(elem?);
            #[cfg(feature = "span")]
            arena.span_info.push(Span::UNDEFINED);
        }
        Ok(arena)
    }

    fn arbitrary_take_rest(u: arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut arena = Self::default();
        for elem in u.arbitrary_take_rest_iter()? {
            arena.set.insert(elem?);
            #[cfg(feature = "span")]
            arena.span_info.push(Span::UNDEFINED);
        }
        Ok(arena)
    }

    #[inline]
    fn size_hint(depth: usize) -> (usize, Option<usize>) {
        let depth_hint = <usize as arbitrary::Arbitrary>::size_hint(depth);
        arbitrary::size_hint::and(depth_hint, (0, None))
    }
}

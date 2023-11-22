use crate::arena::{Arena, Handle, Range, UniqueArena};

type Index = std::num::NonZeroU32;

/// A set of `Handle<T>` values.
pub struct HandleSet<T> {
    /// Bound on zero-based indexes of handles stored in this set.
    len: usize,

    /// `members[i]` is true if the handle with zero-based index `i`
    /// is a member.
    members: bit_set::BitSet,

    /// This type is indexed by values of type `T`.
    as_keys: std::marker::PhantomData<T>,
}

impl<T> HandleSet<T> {
    pub fn for_arena(arena: &impl ArenaType<T>) -> Self {
        let len = arena.len();
        Self {
            len,
            members: bit_set::BitSet::with_capacity(len),
            as_keys: std::marker::PhantomData,
        }
    }

    /// Add `handle` to the set.
    pub fn insert(&mut self, handle: Handle<T>) {
        // Note that, oddly, `Handle::index` does not return a 1-based
        // `Index`, but rather a zero-based `usize`.
        self.members.insert(handle.index());
    }

    /// Add handles from `iter` to the set.
    pub fn insert_iter(&mut self, iter: impl IntoIterator<Item = Handle<T>>) {
        for handle in iter {
            self.insert(handle);
        }
    }

    pub fn contains(&self, handle: Handle<T>) -> bool {
        // Note that, oddly, `Handle::index` does not return a 1-based
        // `Index`, but rather a zero-based `usize`.
        self.members.contains(handle.index())
    }
}

pub trait ArenaType<T> {
    fn len(&self) -> usize;
}

impl<T> ArenaType<T> for Arena<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

impl<T: std::hash::Hash + Eq> ArenaType<T> for UniqueArena<T> {
    fn len(&self) -> usize {
        self.len()
    }
}

/// A map from old handle indices to new, compressed handle indices.
pub struct HandleMap<T> {
    /// The indices assigned to handles in the compacted module.
    ///
    /// If `new_index[i]` is `Some(n)`, then `n` is the 1-based
    /// `Index` of the compacted `Handle` corresponding to the
    /// pre-compacted `Handle` whose zero-based index is `i`. ("Clear
    /// as mud.")
    new_index: Vec<Option<Index>>,

    /// This type is indexed by values of type `T`.
    as_keys: std::marker::PhantomData<T>,
}

impl<T: 'static> HandleMap<T> {
    pub fn from_set(set: HandleSet<T>) -> Self {
        let mut next_index = Index::new(1).unwrap();
        Self {
            new_index: (0..set.len)
                .map(|zero_based_index| {
                    if set.members.contains(zero_based_index) {
                        // This handle will be retained in the compacted version,
                        // so assign it a new index.
                        let this = next_index;
                        next_index = next_index.checked_add(1).unwrap();
                        Some(this)
                    } else {
                        // This handle will be omitted in the compacted version.
                        None
                    }
                })
                .collect(),
            as_keys: std::marker::PhantomData,
        }
    }

    /// Return true if `old` is used in the compacted module.
    pub fn used(&self, old: Handle<T>) -> bool {
        self.new_index[old.index()].is_some()
    }

    /// Return the counterpart to `old` in the compacted module.
    ///
    /// If we thought `old` wouldn't be used in the compacted module, return
    /// `None`.
    pub fn try_adjust(&self, old: Handle<T>) -> Option<Handle<T>> {
        log::trace!(
            "adjusting {} handle [{}] -> [{:?}]",
            std::any::type_name::<T>(),
            old.index() + 1,
            self.new_index[old.index()]
        );
        // Note that `Handle::index` returns a zero-based index,
        // but `Handle::new` accepts a 1-based `Index`.
        self.new_index[old.index()].map(Handle::new)
    }

    /// Return the counterpart to `old` in the compacted module.
    ///
    /// If we thought `old` wouldn't be used in the compacted module, panic.
    pub fn adjust(&self, handle: &mut Handle<T>) {
        *handle = self.try_adjust(*handle).unwrap();
    }

    /// Like `adjust`, but for optional handles.
    pub fn adjust_option(&self, handle: &mut Option<Handle<T>>) {
        if let Some(ref mut handle) = *handle {
            self.adjust(handle);
        }
    }

    /// Shrink `range` to include only used handles.
    ///
    /// Fortunately, compaction doesn't arbitrarily scramble the expressions
    /// in the arena, but instead preserves the order of the elements while
    /// squeezing out unused ones. That means that a contiguous range in the
    /// pre-compacted arena always maps to a contiguous range in the
    /// post-compacted arena. So we just need to adjust the endpoints.
    ///
    /// Compaction may have eliminated the endpoints themselves.
    ///
    /// Use `compacted_arena` to bounds-check the result.
    pub fn adjust_range(&self, range: &mut Range<T>, compacted_arena: &Arena<T>) {
        let mut index_range = range.zero_based_index_range();
        let compacted;
        // Remember that the indices we retrieve from `new_index` are 1-based
        // compacted indices, but the index range we're computing is zero-based
        // compacted indices.
        if let Some(first1) = index_range.find_map(|i| self.new_index[i as usize]) {
            // The first call to `find_map` mutated `index_range` to hold the
            // remainder of original range, which is exactly the range we need
            // to search for the new last handle.
            if let Some(last1) = index_range.rev().find_map(|i| self.new_index[i as usize]) {
                // Build a zero-based end-exclusive range, given one-based handle indices.
                compacted = first1.get() - 1..last1.get();
            } else {
                // The range contains only a single live handle, which
                // we identified with the first `find_map` call.
                compacted = first1.get() - 1..first1.get();
            }
        } else {
            compacted = 0..0;
        };
        *range = Range::from_zero_based_index_range(compacted, compacted_arena);
    }
}

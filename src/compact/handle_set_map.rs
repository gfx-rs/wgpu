use crate::arena::{Arena, Handle, UniqueArena};

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
    ///
    /// Return `true` if the handle was not already in the set. In
    /// other words, return true if it was newly inserted.
    pub fn insert(&mut self, handle: Handle<T>) -> bool {
        // Note that, oddly, `Handle::index` does not return a 1-based
        // `Index`, but rather a zero-based `usize`.
        self.members.insert(handle.index())
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
}

use {RefCount, Stored, WeaklyStored, BufferId, TextureId};
use resource::{BufferUsageFlags, TextureUsageFlags};

use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;
use std::mem;
use std::ops::{BitOr, Range};


#[derive(Clone, Debug, PartialEq)]
#[allow(unused)]
pub enum Tracktion<T> {
    Init,
    Keep,
    Extend { old: T },
    Replace { old: T },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Query<T> {
    pub usage: T,
    pub initialized: bool,
}

bitflags! {
    pub struct TrackPermit: u32 {
        const EXTEND = 1;
        const REPLACE = 2;
    }
}


pub trait GenericUsage {
    fn is_exclusive(&self) -> bool;
}
impl GenericUsage for BufferUsageFlags {
    fn is_exclusive(&self) -> bool {
        BufferUsageFlags::WRITE_ALL.intersects(*self)
    }
}
impl GenericUsage for TextureUsageFlags {
    fn is_exclusive(&self) -> bool {
        TextureUsageFlags::WRITE_ALL.intersects(*self)
    }
}

#[derive(Clone)]
struct Track<U> {
    ref_count: RefCount,
    init: U,
    last: U,
}

unsafe impl<U> Send for Track<U> {}
unsafe impl<U> Sync for Track<U> {}

//TODO: consider having `I` as an associated type of `U`?
pub struct Tracker<I, U> {
    map: HashMap<WeaklyStored<I>, Track<U>>,
}
pub type BufferTracker = Tracker<BufferId, BufferUsageFlags>;
pub type TextureTracker = Tracker<TextureId, TextureUsageFlags>;

impl<
    I: Clone + Hash + Eq,
    U: Copy + GenericUsage + BitOr<Output = U> + PartialEq,
> Tracker<I, U> {
    pub(crate) fn new() -> Self {
        Tracker {
            map: HashMap::new(),
        }
    }

    /// Get the last usage on a resource.
    pub(crate) fn query(
        &mut self, stored: &Stored<I>, default: U
    ) -> Query<U> {
        match self.map.entry(WeaklyStored(stored.value.clone())) {
            Entry::Vacant(e) => {
                e.insert(Track {
                    ref_count: stored.ref_count.clone(),
                    init: default,
                    last: default,
                });
                Query {
                    usage: default,
                    initialized: true,
                }
            }
            Entry::Occupied(e) => {
                Query {
                    usage: e.get().last,
                    initialized: false,
                }
            }
        }
    }

    /// Transit a specified resource into a different usage.
    pub(crate) fn transit(
        &mut self, id: I, ref_count: &RefCount, usage: U, permit: TrackPermit
    ) -> Result<Tracktion<U>, U> {
        match self.map.entry(WeaklyStored(id)) {
            Entry::Vacant(e) => {
                e.insert(Track {
                    ref_count: ref_count.clone(),
                    init: usage,
                    last: usage,
                });
                Ok(Tracktion::Init)
            }
            Entry::Occupied(mut e) => {
                let old = e.get().last;
                if usage == old {
                    Ok(Tracktion::Keep)
                } else if permit.contains(TrackPermit::EXTEND) && !(old | usage).is_exclusive() {
                    e.get_mut().last = old | usage;
                    Ok(Tracktion::Extend { old })
                } else if permit.contains(TrackPermit::REPLACE) {
                    e.get_mut().last = usage;
                    Ok(Tracktion::Replace { old })
                } else {
                    Err(old)
                }
            }
        }
    }

    /// Consume another tacker, adding it's transitions to `self`.
    pub fn consume<'a>(
        &'a mut self, other: &'a Self
    ) -> impl 'a + Iterator<Item = (I, Range<U>)> {
        other.map
            .iter()
            .flat_map(move |(id, new)| match self.map.entry(WeaklyStored(id.0.clone())) {
                Entry::Vacant(e) => {
                    e.insert(new.clone());
                    None
                }
                Entry::Occupied(mut e) => {
                    let old = mem::replace(&mut e.get_mut().last, new.last);
                    if old == new.init {
                        None
                    } else {
                        Some((id.0.clone(), old .. new.last))
                    }
                }
            })
    }

    /// Return an iterator over used resources keys.
    pub fn used<'a>(&'a self) -> impl 'a + Iterator<Item = I> {
        self.map.keys().map(|&WeaklyStored(ref id)| id.clone())
    }
}

use {Stored, BufferId, TextureId};
use resource::{BufferUsageFlags, TextureUsageFlags};

use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;
use std::ops::{BitOr, Range};


#[derive(Clone, Debug, PartialEq)]
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


//TODO: consider having `I` as an associated type of `U`?
pub struct Tracker<I, U> {
    map: HashMap<Stored<I>, Range<U>>,
}
pub type BufferTracker = Tracker<BufferId, BufferUsageFlags>;
pub type TextureTracker = Tracker<TextureId, TextureUsageFlags>;

impl<
    I: Clone + Hash + Eq,
    U: Copy + GenericUsage + BitOr<Output = U> + PartialEq,
> Tracker<I, U> {
    pub fn new() -> Self {
        Tracker {
            map: HashMap::new(),
        }
    }

    pub fn query(&mut self, id: I, default: U) -> Query<U> {
        match self.map.entry(Stored(id)) {
            Entry::Vacant(e) => {
                e.insert(default .. default);
                Query {
                    usage: default,
                    initialized: true,
                }
            }
            Entry::Occupied(e) => {
                Query {
                    usage: e.get().end,
                    initialized: false,
                }
            }
        }
    }

    pub fn transit(&mut self, id: I, usage: U, permit: TrackPermit) -> Result<Tracktion<U>, U> {
        match self.map.entry(Stored(id)) {
            Entry::Vacant(e) => {
                e.insert(usage .. usage);
                Ok(Tracktion::Init)
            }
            Entry::Occupied(mut e) => {
                let old = e.get().end;
                if usage == old {
                    Ok(Tracktion::Keep)
                } else if permit.contains(TrackPermit::EXTEND) && !(old | usage).is_exclusive() {
                    e.get_mut().end = old | usage;
                    Ok(Tracktion::Extend { old })
                } else if permit.contains(TrackPermit::REPLACE) {
                    e.get_mut().end = usage;
                    Ok(Tracktion::Replace { old })
                } else {
                    Err(old)
                }
            }
        }
    }

    pub fn consume<'a>(&'a mut self, other: &'a Self) -> impl 'a + Iterator<Item = (I, Range<U>)> {
        other.map
            .iter()
            .flat_map(move |(id, new)| match self.transit(id.0.clone(), new.end, TrackPermit::REPLACE) {
                Ok(Tracktion::Init) |
                Ok(Tracktion::Keep) => None,
                Ok(Tracktion::Replace { old }) => Some((id.0.clone(), old .. new.end)),
                Ok(Tracktion::Extend { .. }) |
                Err(_) => panic!("Unable to consume a resource transition!"),
            })
    }
}

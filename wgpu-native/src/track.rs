use {Stored, BufferId, TextureId};
use resource::{BufferUsageFlags, TextureUsageFlags};

use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;
use std::ops::BitOr;
use std::sync::Mutex;


#[derive(Clone, Debug, PartialEq)]
pub enum UseAction<T> {
    Init,
    Keep,
    Extend { old: T },
    Replace { old: T },
}

bitflags! {
    pub struct UsePermit: u32 {
        const EXTEND = 1;
        const REPLACE = 2;
    }
}


trait GenericUsage {
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

#[derive(Default)]
pub struct Tracker {
    buffers: Mutex<HashMap<Stored<BufferId>, BufferUsageFlags>>,
    textures: Mutex<HashMap<Stored<TextureId>, TextureUsageFlags>>,
}

impl Tracker {
    fn use_impl<I, U>(
        map: &mut HashMap<I, U>, id: I, usage: U, permit: UsePermit
    ) -> Result<UseAction<U>, U>
    where
        I: Hash + Eq,
        U: Copy + GenericUsage + BitOr<Output = U> + PartialEq,
    {
        match map.entry(id) {
            Entry::Vacant(e) => {
                e.insert(usage);
                Ok(UseAction::Init)
            }
            Entry::Occupied(mut e) => {
                let old = *e.get();
                if usage == old {
                    Ok(UseAction::Keep)
                } else if permit.contains(UsePermit::EXTEND) && !(old | usage).is_exclusive() {
                    Ok(UseAction::Extend { old: e.insert(old | usage) })
                } else if permit.contains(UsePermit::REPLACE) {
                    Ok(UseAction::Replace { old: e.insert(usage) })
                } else {
                    Err(old)
                }
            }
        }
    }

    pub(crate) fn use_buffer(
        &self, id: Stored<BufferId>, usage: BufferUsageFlags, permit: UsePermit,
    ) -> Result<UseAction<BufferUsageFlags>, BufferUsageFlags> {
        Self::use_impl(&mut *self.buffers.lock().unwrap(), id, usage, permit)
    }

    pub(crate) fn use_texture(
        &self, id: Stored<TextureId>, usage: TextureUsageFlags, permit: UsePermit,
    ) -> Result<UseAction<TextureUsageFlags>, TextureUsageFlags> {
        Self::use_impl(&mut *self.textures.lock().unwrap(), id, usage, permit)
    }
}

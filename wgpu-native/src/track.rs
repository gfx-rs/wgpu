use {Stored, BufferId, TextureId};
use resource::{BufferUsageFlags, TextureUsageFlags};

use std::collections::hash_map::{Entry, HashMap};
use std::ops::Range;
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


#[derive(Default)]
pub struct Tracker {
    buffers: Mutex<HashMap<Stored<BufferId>, Range<BufferUsageFlags>>>,
    textures: Mutex<HashMap<Stored<TextureId>, Range<TextureUsageFlags>>>,
}

impl Tracker {
    pub(crate) fn use_buffer(
        &self, id: Stored<BufferId>, usage: BufferUsageFlags, permit: UsePermit,
    ) -> Result<UseAction<BufferUsageFlags>, BufferUsageFlags> {
        match self.buffers.lock().unwrap().entry(id) {
            Entry::Vacant(e) => {
                e.insert(usage .. usage);
                Ok(UseAction::Init)
            }
            Entry::Occupied(mut e) => {
                let old = e.get().end;
                if usage == old {
                    Ok(UseAction::Keep)
                } else if permit.contains(UsePermit::EXTEND) &&
                    !BufferUsageFlags::WRITE_ALL.intersects(old | usage)
                {
                    e.get_mut().end |= usage;
                    Ok(UseAction::Extend { old })
                } else if permit.contains(UsePermit::REPLACE) {
                    e.get_mut().end = usage;
                    Ok(UseAction::Replace { old })
                } else {
                    Err(old)
                }
            }
        }
    }

    pub(crate) fn use_texture(
        &self, id: Stored<TextureId>, usage: TextureUsageFlags, permit: UsePermit,
    ) -> Result<UseAction<TextureUsageFlags>, TextureUsageFlags> {
        match self.textures.lock().unwrap().entry(id) {
            Entry::Vacant(e) => {
                e.insert(usage .. usage);
                Ok(UseAction::Init)
            }
            Entry::Occupied(mut e) => {
                let old = e.get().end;
                if usage == old {
                    Ok(UseAction::Keep)
                } else if permit.contains(UsePermit::EXTEND) &&
                    !TextureUsageFlags::WRITE_ALL.intersects(old | usage)
                {
                    e.get_mut().end |= usage;
                    Ok(UseAction::Extend { old })
                } else if permit.contains(UsePermit::REPLACE) {
                    e.get_mut().end = usage;
                    Ok(UseAction::Replace { old })
                } else {
                    Err(old)
                }
            }
        }
    }
}

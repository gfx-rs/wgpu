use hal::backend::FastHashMap;
use parking_lot::{Mutex, MutexGuard};
use std::sync::Arc;


pub type Id = u32;
pub type ItemsGuard<'a, T> = MutexGuard<'a, Items<T>>;

pub struct Items<T> {
    next_id: Id,
    tracked: FastHashMap<Id, T>,
    free: Vec<Id>,
}

impl<T> Items<T> {
    fn new() -> Self {
        Items {
            next_id: 0,
            tracked: FastHashMap::default(),
            free: Vec::new(),
        }
    }
}

impl<T> super::Items<T> for Items<T> {
    fn register(&mut self, handle: T) -> Id {
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                self.next_id += 1;
                self.next_id - 1
            }
        };
        self.tracked.insert(id, handle);
        id
    }

    fn get(&self, id: Id) -> super::Item<T> {
        self.tracked.get(&id).unwrap()
    }

    fn get_mut(&mut self, id: Id) -> super::ItemMut<T> {
        self.tracked.get_mut(&id).unwrap()
    }

    fn take(&mut self, id: Id) -> T {
        self.free.push(id);
        self.tracked.remove(&id).unwrap()
    }
}

pub struct Registry<T> {
    items: Arc<Mutex<Items<T>>>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Registry {
            items: Arc::new(Mutex::new(Items::new())),
        }
    }
}

impl<T> super::Registry<T> for Registry<T> {
    fn lock(&self) -> ItemsGuard<T> {
        self.items.lock()
    }
}

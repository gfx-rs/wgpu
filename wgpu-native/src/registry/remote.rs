use hal::backend::FastHashMap;

pub type Id = u32;

pub struct Items<T> {
    next_id: Id,
    tracked: FastHashMap<Id, T>,
    free: Vec<Id>,
}

impl<T> Default for Items<T> {
    fn default() -> Self {
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

    fn get(&self, id: Id) -> &T {
        self.tracked.get(&id).unwrap()
    }

    fn get_mut(&mut self, id: Id) -> &mut T {
        self.tracked.get_mut(&id).unwrap()
    }

    fn take(&mut self, id: Id) -> T {
        self.free.push(id);
        self.tracked.remove(&id).unwrap()
    }
}

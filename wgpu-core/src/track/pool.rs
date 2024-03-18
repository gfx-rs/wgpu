use bit_vec::BitVec;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
pub struct VecPool {
    pool: OnceCell<Mutex<Vec<Vec<()>>>>,
}

impl VecPool {
    pub const fn new() -> Self {
        Self{ pool: OnceCell::new() }
    }

    /// safety: a pool instance must only ever be called with a single type T to ensure transmute is sound
    /// (`#[repr(rust)]` does not guarantee equal member order for different monomorphs)
    pub unsafe fn get<T>(&self) -> Vec<T> {
        unsafe { std::mem::transmute(self.pool.get_or_init(|| Default::default()).lock().pop().unwrap_or_default()) }
    }

    /// safety: a pool instance must only ever be called with a single type T to ensure transmute is sound
    /// (`#[repr(rust)]` does not guarantee equal member order for different monomorphs)
    pub unsafe fn put<T>(&self, vec: &mut Vec<T>) {
        let mut vec = std::mem::take(vec);
        vec.clear();
        let vec = unsafe { std::mem::transmute(vec) };
        self.pool.get().unwrap().lock().push(vec);
    }
}

pub struct BitvecPool {
    pool: OnceCell<Mutex<Vec<BitVec<usize>>>>,
}

impl BitvecPool {
    pub const fn new() -> Self {
        Self{ pool: OnceCell::new() }
    }

    pub fn get(&self) -> BitVec<usize> {
        self.pool.get_or_init(|| Default::default()).lock().pop().unwrap_or_default()
    }

    pub fn put(&self, vec: &mut BitVec<usize>) {
        let mut vec = std::mem::take(vec);
        vec.clear();
        self.pool.get().unwrap().lock().push(vec);
    }
}

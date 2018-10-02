use std::marker::PhantomData;
use std::os::raw::c_void;


pub type Id = *mut c_void;
pub type ItemsGuard<'a, T> = Items<T>;

pub struct Items<T> {
    marker: PhantomData<T>,
}

impl<T> super::Items<T> for Items<T> {
    fn register(&mut self, handle: T) -> Id {
        Box::into_raw(Box::new(handle)) as *mut _ as *mut c_void
    }

    fn get(&self, id: Id) -> super::Item<T> {
        unsafe { (id as *mut T).as_ref() }.unwrap()
    }

    fn get_mut(&mut self, id: Id) -> super::ItemMut<T> {
        unsafe { (id as *mut T).as_mut() }.unwrap()
    }

    fn take(&mut self, id: Id) -> T {
        unsafe { *Box::from_raw(id as *mut T) }
    }
}

pub struct Registry<T> {
    marker: PhantomData<T>,
}

impl<T> Default for Registry<T> {
    fn default() -> Self {
        Registry {
            marker: PhantomData,
        }
    }
}

impl<T> super::Registry<T> for Registry<T> {
    fn lock(&self) -> ItemsGuard<T> {
        Items {
            marker: PhantomData,
        }
    }
}

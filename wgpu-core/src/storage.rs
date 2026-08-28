use alloc::sync::Arc;

use crate::id::Marker;
use crate::resource::ResourceType;
use wgpu_sync::Mutex;

/// Not a public API. For use only by `player`.
#[doc(hidden)]
pub trait StorageItem: ResourceType {
    type Marker: Marker;
}

impl<T: ResourceType> ResourceType for Arc<T> {
    const TYPE: &'static str = T::TYPE;
}

impl<T: StorageItem> StorageItem for Arc<T> {
    type Marker = T::Marker;
}

impl<T: ResourceType> ResourceType for Mutex<T> {
    const TYPE: &'static str = T::TYPE;
}

impl<T: StorageItem> StorageItem for Mutex<T> {
    type Marker = T::Marker;
}

#[macro_export]
macro_rules! impl_storage_item {
    ($ty:ident) => {
        impl $crate::storage::StorageItem for $ty {
            type Marker = $crate::id::markers::$ty;
        }
    };
}

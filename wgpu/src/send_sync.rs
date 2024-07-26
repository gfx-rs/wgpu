use std::any::Any;
use std::fmt;

use wgt::WasmNotSendSync;

pub trait AnyWasmNotSendSync: Any + WasmNotSendSync {
    fn upcast_any_ref(&self) -> &dyn Any;
}
impl<T: Any + WasmNotSendSync> AnyWasmNotSendSync for T {
    #[inline]
    fn upcast_any_ref(&self) -> &dyn Any {
        self
    }
}

impl dyn AnyWasmNotSendSync + 'static {
    #[inline]
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.upcast_any_ref().downcast_ref::<T>()
    }
}

impl fmt::Debug for dyn AnyWasmNotSendSync {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Any").finish_non_exhaustive()
    }
}

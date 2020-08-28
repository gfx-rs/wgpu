//! Functions which export shader modules into binary and text formats.

#[cfg(feature = "glsl-out")]
pub mod glsl;
pub mod msl;
#[cfg(feature = "spirv-out")]
pub mod spv;

#[derive(Debug)]
pub enum MaybeOwned<'a, T: 'a> {
    Borrowed(&'a T),
    Owned(T),
}

impl<T> MaybeOwned<'_, T> {
    fn borrow(&self) -> &T {
        match *self {
            MaybeOwned::Borrowed(inner) => inner,
            MaybeOwned::Owned(ref inner) => inner,
        }
    }
}

pub type BorrowType<'a> = MaybeOwned<'a, crate::TypeInner>;

impl crate::Module {
    fn borrow_type(&self, handle: crate::Handle<crate::Type>) -> BorrowType {
        MaybeOwned::Borrowed(&self.types[handle].inner)
    }
}

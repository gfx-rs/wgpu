use alloc::borrow::Cow;

#[cfg(feature = "std")]
use std::path::Path;

#[cfg(not(feature = "std"))]
use alloc::string::String;

pub trait AsPath {
    #[cfg(feature = "std")]
    fn as_ref(&self) -> &Path;

    fn to_string_lossy(&self) -> Cow<'_, str>;
}

#[cfg(feature = "std")]
impl<T: AsRef<Path> + ?Sized> AsPath for T {
    fn as_ref(&self) -> &Path {
        self.as_ref()
    }

    fn to_string_lossy(&self) -> Cow<'_, str> {
        self.as_ref().to_string_lossy()
    }
}

#[cfg(not(feature = "std"))]
impl AsPath for String {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(self.as_str())
    }
}

#[cfg(not(feature = "std"))]
impl AsPath for str {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self)
    }
}

#[cfg(not(feature = "std"))]
impl<T: AsPath + ?Sized> AsPath for &T {
    fn to_string_lossy(&self) -> Cow<'_, str> {
        (*self).to_string_lossy()
    }
}

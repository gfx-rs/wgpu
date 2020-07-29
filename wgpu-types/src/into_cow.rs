use std::borrow::Cow;

/// Implements conversion into a cow.
pub trait IntoCow<'c, CowType: ?Sized + ToOwned> {
    fn into_cow(self) -> Cow<'c, CowType>;
}

impl<'c, T: Clone> IntoCow<'c, [T]> for &'c [T] {
    fn into_cow(self) -> Cow<'c, [T]> {
        Cow::Borrowed(self)
    }
}

impl<'c> IntoCow<'c, str> for &'c str {
    fn into_cow(self) -> Cow<'c, str> {
        Cow::Borrowed(self)
    }
}

impl<T: Clone> IntoCow<'static, [T]> for Vec<T> {
    fn into_cow(self) -> Cow<'static, [T]> {
        Cow::Owned(self)
    }
}

impl IntoCow<'static, str> for String {
    fn into_cow(self) -> Cow<'static, str> {
        Cow::Owned(self)
    }
}

macro_rules! into_cow_array {
    ($($number:literal),*) => {$(
        impl<'c, T: Clone> IntoCow<'c, [T]> for &'c [T; $number] {
            fn into_cow(self) -> Cow<'c, [T]> {
                Cow::Borrowed(&self[..])
            }
        }
    )*};
}

into_cow_array!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32
);

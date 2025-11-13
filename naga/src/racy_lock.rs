use alloc::boxed::Box;
use once_cell::race::OnceBox;

/// An alternative to [`LazyLock`] based on [`OnceBox`].
///
/// [`LazyLock`]: https://doc.rust-lang.org/stable/std/sync/struct.LazyLock.html
pub struct RacyLock<T: 'static> {
    inner: OnceBox<T>,
    init: fn() -> T,
}

impl<T: 'static> RacyLock<T> {
    /// Creates a new [`RacyLock`], which will initialize using the provided `init` function.
    pub const fn new(init: fn() -> T) -> Self {
        Self {
            inner: OnceBox::new(),
            init,
        }
    }
}

impl<T: 'static> core::ops::Deref for RacyLock<T> {
    type Target = T;

    /// Loads the internal value, initializing it if required.
    fn deref(&self) -> &Self::Target {
        self.inner.get_or_init(|| Box::new((self.init)()))
    }
}

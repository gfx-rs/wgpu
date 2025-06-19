#![cfg_attr(
    not(any(glsl_out, hlsl_out, msl_out, feature = "wgsl-in", wgsl_out)),
    expect(
        dead_code,
        reason = "RacyLock is only required for the above configurations"
    )
)]

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

    /// Loads the internal value, initializing it if required.
    pub fn get(&self) -> &T {
        self.inner.get_or_init(|| Box::new((self.init)()))
    }
}

impl<T: 'static> core::ops::Deref for RacyLock<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

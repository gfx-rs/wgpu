/// Token of the user agreeing to access experimental features.
#[derive(Debug, Default, Copy, Clone)]
pub struct ExperimentalFeatures {
    enabled: bool,
}

impl ExperimentalFeatures {
    /// Uses of [`Features`] prefixed with "EXPERIMENTAL" are disallowed.
    ///
    /// [`Features`]: ../wgpu/struct.Features.html
    pub const fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Uses of [`Features`] prefixed with "EXPERIMENTAL" may result
    /// in undefined behavior when used incorrectly. The exact bounds
    /// of these issues varies by the feature. These instances are
    /// inherently bugs in our implementation that we will eventually fix.
    ///
    /// By giving access to still work-in-progress APIs, users can get
    /// access to newer technology sooner, and we can work with users
    /// to fix bugs quicker.
    ///
    /// Look inside our repo at the [`api-specs`] for more information
    /// on various experimental apis.
    ///
    /// # Safety
    ///
    /// - You acknowledge that there may be UB-containing bugs in these
    ///   apis and those may be hit by calling otherwise safe code.
    /// - You agree to report any such bugs to us, if you find them.
    ///
    /// [`Features`]: ../wgpu/struct.Features.html
    /// [`api-specs`]: https://github.com/gfx-rs/wgpu/tree/trunk/docs/api-specs
    pub const unsafe fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Returns true if the user has agreed to access experimental features.
    pub const fn is_enabled(&self) -> bool {
        self.enabled
    }
}

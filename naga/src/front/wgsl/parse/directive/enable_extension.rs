use crate::{front::wgsl::error::Error, Span};

/// If `Some`, indicates that `enable f16;` was written at the given [`Span`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnableExtensions {
    f16: bool,
}

impl EnableExtensions {
    pub(crate) const fn empty() -> Self {
        Self { f16: false }
    }

    pub(crate) fn add(&mut self, ext: ImplementedEnableExtension) {
        let field = match ext {
            ImplementedEnableExtension::F16 => &mut self.f16,
        };
        *field = true;
    }

    pub(crate) const fn contains(&self, ext: ImplementedEnableExtension) -> bool {
        match ext {
            ImplementedEnableExtension::F16 => self.f16,
        }
    }
}

impl Default for EnableExtensions {
    fn default() -> Self {
        Self::empty()
    }
}

/// A shader language extension not guaranteed to be present in all environments.
///
/// WGSL spec.: <https://www.w3.org/TR/WGSL/#enable-extensions-sec>
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum EnableExtension {
    Implemented(ImplementedEnableExtension),
    Unimplemented(UnimplementedEnableExtension),
}

impl EnableExtension {
    pub(crate) fn from_ident(word: &str, span: Span) -> Result<(Self, &'static str), Error<'_>> {
        match word {
            "f16" => Ok((Self::Implemented(ImplementedEnableExtension::F16), "f16")),
            "clip_distances" => Ok((
                Self::Unimplemented(UnimplementedEnableExtension::ClipDistances),
                "clip_distances",
            )),
            "dual_source_blending" => Ok((
                Self::Unimplemented(UnimplementedEnableExtension::DualSourceBlending),
                "dual_source_blending",
            )),
            _ => Err(Error::UnknownEnableExtension(span, word)),
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ImplementedEnableExtension {
    /// Enables `f16`/`half` primitive support in all shader languages.
    ///
    /// In the WGSL standard, this corresponds to [`enable f16;`].
    ///
    /// [`enable f16;`]: https://www.w3.org/TR/WGSL/#extension-f16
    F16,
}

#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum UnimplementedEnableExtension {
    /// Enables the `clip_distances` variable in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable clip_distances;`].
    ///
    /// [`enable clip_distances;`]: https://www.w3.org/TR/WGSL/#extension-f16
    ClipDistances,
    /// Enables the `blend_src` attribute in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable dual_source_blending;`].
    ///
    /// [`enable dual_source_blending;`]: https://www.w3.org/TR/WGSL/#extension-f16
    DualSourceBlending,
}

impl UnimplementedEnableExtension {
    pub(crate) const fn tracking_issue(self) -> u16 {
        match self {
            Self::ClipDistances => 6236,
            Self::DualSourceBlending => 6402,
        }
    }
}

//! `enable …;` extensions in WGSL.
//!
//! The focal point of this module is the [`EnableExtension`] API.

use crate::front::wgsl::{Error, Result};
use crate::Span;

use alloc::boxed::Box;

/// Tracks the status of every enable-extension known to Naga.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnableExtensions {
    dual_source_blending: bool,
    /// Whether `enable f16;` was written earlier in the shader module.
    f16: bool,
    clip_distances: bool,
}

impl EnableExtensions {
    pub(crate) const fn empty() -> Self {
        Self {
            f16: false,
            dual_source_blending: false,
            clip_distances: false,
        }
    }

    /// Add an enable-extension to the set requested by a module.
    pub(crate) fn add(&mut self, ext: ImplementedEnableExtension) {
        let field = match ext {
            ImplementedEnableExtension::DualSourceBlending => &mut self.dual_source_blending,
            ImplementedEnableExtension::F16 => &mut self.f16,
            ImplementedEnableExtension::ClipDistances => &mut self.clip_distances,
        };
        *field = true;
    }

    /// Query whether an enable-extension tracked here has been requested.
    pub(crate) const fn contains(&self, ext: ImplementedEnableExtension) -> bool {
        match ext {
            ImplementedEnableExtension::DualSourceBlending => self.dual_source_blending,
            ImplementedEnableExtension::F16 => self.f16,
            ImplementedEnableExtension::ClipDistances => self.clip_distances,
        }
    }
}

impl Default for EnableExtensions {
    fn default() -> Self {
        Self::empty()
    }
}

/// An enable-extension not guaranteed to be present in all environments.
///
/// WGSL spec.: <https://www.w3.org/TR/WGSL/#enable-extensions-sec>
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum EnableExtension {
    Implemented(ImplementedEnableExtension),
    Unimplemented(UnimplementedEnableExtension),
}

impl From<ImplementedEnableExtension> for EnableExtension {
    fn from(value: ImplementedEnableExtension) -> Self {
        Self::Implemented(value)
    }
}

impl EnableExtension {
    const F16: &'static str = "f16";
    const CLIP_DISTANCES: &'static str = "clip_distances";
    const DUAL_SOURCE_BLENDING: &'static str = "dual_source_blending";
    const SUBGROUPS: &'static str = "subgroups";

    /// Convert from a sentinel word in WGSL into its associated [`EnableExtension`], if possible.
    pub(crate) fn from_ident(word: &str, span: Span) -> Result<'_, Self> {
        Ok(match word {
            Self::F16 => Self::Implemented(ImplementedEnableExtension::F16),
            Self::CLIP_DISTANCES => Self::Implemented(ImplementedEnableExtension::ClipDistances),
            Self::DUAL_SOURCE_BLENDING => {
                Self::Implemented(ImplementedEnableExtension::DualSourceBlending)
            }
            Self::SUBGROUPS => Self::Unimplemented(UnimplementedEnableExtension::Subgroups),
            _ => return Err(Box::new(Error::UnknownEnableExtension(span, word))),
        })
    }

    /// Maps this [`EnableExtension`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::Implemented(kind) => match kind {
                ImplementedEnableExtension::DualSourceBlending => Self::DUAL_SOURCE_BLENDING,
                ImplementedEnableExtension::F16 => Self::F16,
                ImplementedEnableExtension::ClipDistances => Self::CLIP_DISTANCES,
            },
            Self::Unimplemented(kind) => match kind {
                UnimplementedEnableExtension::Subgroups => Self::SUBGROUPS,
            },
        }
    }
}

/// A variant of [`EnableExtension::Implemented`].
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum ImplementedEnableExtension {
    /// Enables `f16`/`half` primitive support in all shader languages.
    ///
    /// In the WGSL standard, this corresponds to [`enable f16;`].
    ///
    /// [`enable f16;`]: https://www.w3.org/TR/WGSL/#extension-f16
    F16,
    /// Enables the `blend_src` attribute in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable dual_source_blending;`].
    ///
    /// [`enable dual_source_blending;`]: https://www.w3.org/TR/WGSL/#extension-dual_source_blending
    DualSourceBlending,
    /// Enables the `clip_distances` variable in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable clip_distances;`].
    ///
    /// [`enable clip_distances;`]: https://www.w3.org/TR/WGSL/#extension-clip_distances
    ClipDistances,
}

/// A variant of [`EnableExtension::Unimplemented`].
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum UnimplementedEnableExtension {
    /// Enables subgroup built-ins in all languages.
    ///
    /// In the WGSL standard, this corresponds to [`enable subgroups;`].
    ///
    /// [`enable subgroups;`]: https://www.w3.org/TR/WGSL/#extension-subgroups
    Subgroups,
}

impl UnimplementedEnableExtension {
    pub(crate) const fn tracking_issue_num(self) -> u16 {
        match self {
            Self::Subgroups => 5555,
        }
    }
}

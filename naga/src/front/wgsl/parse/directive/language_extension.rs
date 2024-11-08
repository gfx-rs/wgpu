//! `requires …;` extensions in WGSL.
//!
//! The focal point of this module is the [`LanguageExtension`] API.

/// A language extension not guaranteed to be present in all environments.
///
/// WGSL spec.: <https://www.w3.org/TR/WGSL/#language-extensions-sec>
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum LanguageExtension {
    #[allow(unused)]
    Implemented(ImplementedLanguageExtension),
    Unimplemented(UnimplementedLanguageExtension),
}

impl LanguageExtension {
    const READONLY_AND_READWRITE_STORAGE_TEXTURES: &'static str =
        "readonly_and_readwrite_storage_textures";
    const PACKED4X8_INTEGER_DOT_PRODUCT: &'static str = "packed_4x8_integer_dot_product";
    const UNRESTRICTED_POINTER_PARAMETERS: &'static str = "unrestricted_pointer_parameters";
    const POINTER_COMPOSITE_ACCESS: &'static str = "pointer_composite_access";

    /// Convert from a sentinel word in WGSL into its associated [`LanguageExtension`], if possible.
    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::READONLY_AND_READWRITE_STORAGE_TEXTURES => Self::Unimplemented(
                UnimplementedLanguageExtension::ReadOnlyAndReadWriteStorageTextures,
            ),
            Self::PACKED4X8_INTEGER_DOT_PRODUCT => {
                Self::Unimplemented(UnimplementedLanguageExtension::Packed4x8IntegerDotProduct)
            }
            Self::UNRESTRICTED_POINTER_PARAMETERS => {
                Self::Unimplemented(UnimplementedLanguageExtension::UnrestrictedPointerParameters)
            }
            Self::POINTER_COMPOSITE_ACCESS => {
                Self::Unimplemented(UnimplementedLanguageExtension::PointerCompositeAccess)
            }
            _ => return None,
        })
    }

    /// Maps this [`LanguageExtension`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::Implemented(kind) => match kind {},
            Self::Unimplemented(kind) => match kind {
                UnimplementedLanguageExtension::ReadOnlyAndReadWriteStorageTextures => {
                    Self::READONLY_AND_READWRITE_STORAGE_TEXTURES
                }
                UnimplementedLanguageExtension::Packed4x8IntegerDotProduct => {
                    Self::PACKED4X8_INTEGER_DOT_PRODUCT
                }
                UnimplementedLanguageExtension::UnrestrictedPointerParameters => {
                    Self::UNRESTRICTED_POINTER_PARAMETERS
                }
                UnimplementedLanguageExtension::PointerCompositeAccess => {
                    Self::POINTER_COMPOSITE_ACCESS
                }
            },
        }
    }
}

/// A variant of [`LanguageExtension::Implemented`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum ImplementedLanguageExtension {}

/// A variant of [`LanguageExtension::Unimplemented`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum UnimplementedLanguageExtension {
    ReadOnlyAndReadWriteStorageTextures,
    Packed4x8IntegerDotProduct,
    UnrestrictedPointerParameters,
    PointerCompositeAccess,
}

impl UnimplementedLanguageExtension {
    pub(crate) const fn tracking_issue_num(self) -> u16 {
        match self {
            Self::ReadOnlyAndReadWriteStorageTextures => 6204,
            Self::Packed4x8IntegerDotProduct => 6445,
            Self::UnrestrictedPointerParameters => 5158,
            Self::PointerCompositeAccess => 6192,
        }
    }
}

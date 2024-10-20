//! WGSL directives. The focal point of this API is [`DirectiveKind`].
//!
//! See also <https://www.w3.org/TR/WGSL/#directives>.

/// A parsed sentinel word indicating the type of directive to be parsed next.
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
pub enum DirectiveKind {
    Unimplemented(UnimplementedDirectiveKind),
}

impl DirectiveKind {
    const DIAGNOSTIC: &'static str = "diagnostic";
    const ENABLE: &'static str = "enable";
    const REQUIRES: &'static str = "requires";

    /// Convert from a sentinel word in WGSL into its associated [`DirectiveKind`], if possible.
    pub fn from_ident(s: &str) -> Option<Self> {
        Some(match s {
            Self::DIAGNOSTIC => Self::Unimplemented(UnimplementedDirectiveKind::Diagnostic),
            Self::ENABLE => Self::Unimplemented(UnimplementedDirectiveKind::Enable),
            Self::REQUIRES => Self::Unimplemented(UnimplementedDirectiveKind::Requires),
            _ => return None,
        })
    }

    /// Maps this [`DirectiveKind`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::Unimplemented(kind) => match kind {
                UnimplementedDirectiveKind::Diagnostic => Self::DIAGNOSTIC,
                UnimplementedDirectiveKind::Enable => Self::ENABLE,
                UnimplementedDirectiveKind::Requires => Self::REQUIRES,
            },
        }
    }

    #[cfg(test)]
    fn iter() -> impl Iterator<Item = Self> {
        use strum::IntoEnumIterator;

        UnimplementedDirectiveKind::iter().map(Self::Unimplemented)
    }
}

/// A [`DirectiveKind`] that is not yet implemented. See [`DirectiveKind::Unimplemented`].
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(test, derive(strum::EnumIter))]
pub enum UnimplementedDirectiveKind {
    Diagnostic,
    Enable,
    Requires,
}

impl UnimplementedDirectiveKind {
    pub const fn tracking_issue_num(self) -> u16 {
        match self {
            Self::Diagnostic => 5320,
            Self::Requires => 6350,
            Self::Enable => 5476,
        }
    }
}

#[cfg(test)]
mod test {
    use strum::IntoEnumIterator;

    use crate::front::wgsl::assert_parse_err;

    use super::{DirectiveKind, UnimplementedDirectiveKind};

    #[test]
    fn unimplemented_directives() {
        for unsupported_shader in UnimplementedDirectiveKind::iter() {
            let shader;
            let expected_msg;
            match unsupported_shader {
                UnimplementedDirectiveKind::Diagnostic => {
                    shader = "diagnostic(off,derivative_uniformity);";
                    expected_msg = "\
error: `diagnostic` is not yet implemented
  ┌─ wgsl:1:1
  │
1 │ diagnostic(off,derivative_uniformity);
  │ ^^^^^^^^^^ this global directive is standard, but not yet implemented
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/5320>, so they can prioritize it!

";
                }
                UnimplementedDirectiveKind::Enable => {
                    shader = "enable f16;";
                    expected_msg = "\
error: `enable` is not yet implemented
  ┌─ wgsl:1:1
  │
1 │ enable f16;
  │ ^^^^^^ this global directive is standard, but not yet implemented
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/5476>, so they can prioritize it!

";
                }
                UnimplementedDirectiveKind::Requires => {
                    shader = "requires readonly_and_readwrite_storage_textures";
                    expected_msg = "\
error: `requires` is not yet implemented
  ┌─ wgsl:1:1
  │
1 │ requires readonly_and_readwrite_storage_textures
  │ ^^^^^^^^ this global directive is standard, but not yet implemented
  │
  = note: Let Naga maintainers know that you ran into this at <https://github.com/gfx-rs/wgpu/issues/6350>, so they can prioritize it!

";
                }
            };

            assert_parse_err(shader, expected_msg);
        }
    }

    #[test]
    fn directive_after_global_decl() {
        for unsupported_shader in DirectiveKind::iter() {
            let directive;
            let expected_msg;
            match unsupported_shader {
                DirectiveKind::Unimplemented(UnimplementedDirectiveKind::Diagnostic) => {
                    directive = "diagnostic(off,derivative_uniformity)";
                    expected_msg = "\
error: expected global declaration, but found a global directive
  ┌─ wgsl:2:1
  │
2 │ diagnostic(off,derivative_uniformity);
  │ ^^^^^^^^^^ written after first global declaration
  │
  = note: global directives are only allowed before global declarations; maybe hoist this closer to the top of the shader module?

";
                }
                DirectiveKind::Unimplemented(UnimplementedDirectiveKind::Enable) => {
                    directive = "enable f16";
                    expected_msg = "\
error: expected global declaration, but found a global directive
  ┌─ wgsl:2:1
  │
2 │ enable f16;
  │ ^^^^^^ written after first global declaration
  │
  = note: global directives are only allowed before global declarations; maybe hoist this closer to the top of the shader module?

";
                }
                DirectiveKind::Unimplemented(UnimplementedDirectiveKind::Requires) => {
                    directive = "requires readonly_and_readwrite_storage_textures";
                    expected_msg = "\
error: expected global declaration, but found a global directive
  ┌─ wgsl:2:1
  │
2 │ requires readonly_and_readwrite_storage_textures;
  │ ^^^^^^^^ written after first global declaration
  │
  = note: global directives are only allowed before global declarations; maybe hoist this closer to the top of the shader module?

";
                }
            }

            let shader = format!(
                "\
@group(0) @binding(0) var<storage> thing: i32;
{directive};
"
            );
            assert_parse_err(&shader, expected_msg);
        }
    }
}

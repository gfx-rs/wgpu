//! `enable …;` extensions in WGSL.
//!
//! The focal point of this module is the [`EnableExtension`] API.

use crate::front::wgsl::{Error, Result};
use crate::Span;

use alloc::boxed::Box;

/// Tracks the status of every enable-extension known to Naga.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct EnableExtensions {
    wgpu_mesh_shader: bool,
    wgpu_ray_query: bool,
    wgpu_ray_query_vertex_return: bool,
    dual_source_blending: bool,
    /// Whether `enable f16;` was written earlier in the shader module.
    f16: bool,
    clip_distances: bool,
    wgpu_cooperative_matrix: bool,
}

impl EnableExtensions {
    pub(crate) const fn empty() -> Self {
        Self {
            wgpu_mesh_shader: false,
            wgpu_ray_query: false,
            wgpu_ray_query_vertex_return: false,
            f16: false,
            dual_source_blending: false,
            clip_distances: false,
            wgpu_cooperative_matrix: false,
        }
    }

    /// Add an enable-extension to the set requested by a module.
    pub(crate) const fn add(&mut self, ext: ImplementedEnableExtension) {
        let field = match ext {
            ImplementedEnableExtension::WgpuMeshShader => &mut self.wgpu_mesh_shader,
            ImplementedEnableExtension::WgpuRayQuery => &mut self.wgpu_ray_query,
            ImplementedEnableExtension::WgpuRayQueryVertexReturn => {
                &mut self.wgpu_ray_query_vertex_return
            }
            ImplementedEnableExtension::DualSourceBlending => &mut self.dual_source_blending,
            ImplementedEnableExtension::F16 => &mut self.f16,
            ImplementedEnableExtension::ClipDistances => &mut self.clip_distances,
            ImplementedEnableExtension::WgpuCooperativeMatrix => &mut self.wgpu_cooperative_matrix,
        };
        *field = true;
    }

    /// Query whether an enable-extension tracked here has been requested.
    pub(crate) const fn contains(&self, ext: ImplementedEnableExtension) -> bool {
        match ext {
            ImplementedEnableExtension::WgpuMeshShader => self.wgpu_mesh_shader,
            ImplementedEnableExtension::WgpuRayQuery => self.wgpu_ray_query,
            ImplementedEnableExtension::WgpuRayQueryVertexReturn => {
                self.wgpu_ray_query_vertex_return
            }
            ImplementedEnableExtension::DualSourceBlending => self.dual_source_blending,
            ImplementedEnableExtension::F16 => self.f16,
            ImplementedEnableExtension::ClipDistances => self.clip_distances,
            ImplementedEnableExtension::WgpuCooperativeMatrix => self.wgpu_cooperative_matrix,
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
    const MESH_SHADER: &'static str = "wgpu_mesh_shader";
    const RAY_QUERY: &'static str = "wgpu_ray_query";
    const RAY_QUERY_VERTEX_RETURN: &'static str = "wgpu_ray_query_vertex_return";
    const COOPERATIVE_MATRIX: &'static str = "wgpu_cooperative_matrix";
    const SUBGROUPS: &'static str = "subgroups";
    const PRIMITIVE_INDEX: &'static str = "primitive_index";

    /// Convert from a sentinel word in WGSL into its associated [`EnableExtension`], if possible.
    pub(crate) fn from_ident(word: &str, span: Span) -> Result<'_, Self> {
        Ok(match word {
            Self::F16 => Self::Implemented(ImplementedEnableExtension::F16),
            Self::CLIP_DISTANCES => Self::Implemented(ImplementedEnableExtension::ClipDistances),
            Self::DUAL_SOURCE_BLENDING => {
                Self::Implemented(ImplementedEnableExtension::DualSourceBlending)
            }
            Self::MESH_SHADER => Self::Implemented(ImplementedEnableExtension::WgpuMeshShader),
            Self::RAY_QUERY => Self::Implemented(ImplementedEnableExtension::WgpuRayQuery),
            Self::RAY_QUERY_VERTEX_RETURN => {
                Self::Implemented(ImplementedEnableExtension::WgpuRayQueryVertexReturn)
            }
            Self::COOPERATIVE_MATRIX => {
                Self::Implemented(ImplementedEnableExtension::WgpuCooperativeMatrix)
            }
            Self::SUBGROUPS => Self::Unimplemented(UnimplementedEnableExtension::Subgroups),
            Self::PRIMITIVE_INDEX => {
                Self::Unimplemented(UnimplementedEnableExtension::PrimitiveIndex)
            }
            _ => return Err(Box::new(Error::UnknownEnableExtension(span, word))),
        })
    }

    /// Maps this [`EnableExtension`] into the sentinel word associated with it in WGSL.
    pub const fn to_ident(self) -> &'static str {
        match self {
            Self::Implemented(kind) => match kind {
                ImplementedEnableExtension::WgpuMeshShader => Self::MESH_SHADER,
                ImplementedEnableExtension::WgpuRayQuery => Self::RAY_QUERY,
                ImplementedEnableExtension::WgpuRayQueryVertexReturn => {
                    Self::RAY_QUERY_VERTEX_RETURN
                }
                ImplementedEnableExtension::WgpuCooperativeMatrix => Self::COOPERATIVE_MATRIX,
                ImplementedEnableExtension::DualSourceBlending => Self::DUAL_SOURCE_BLENDING,
                ImplementedEnableExtension::F16 => Self::F16,
                ImplementedEnableExtension::ClipDistances => Self::CLIP_DISTANCES,
            },
            Self::Unimplemented(kind) => match kind {
                UnimplementedEnableExtension::Subgroups => Self::SUBGROUPS,
                UnimplementedEnableExtension::PrimitiveIndex => Self::PRIMITIVE_INDEX,
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
    /// Enables the `wgpu_mesh_shader` extension, native only
    WgpuMeshShader,
    /// Enables the `wgpu_ray_query` extension, native only.
    WgpuRayQuery,
    /// Enables the `wgpu_ray_query_vertex_return` extension, native only.
    WgpuRayQueryVertexReturn,
    /// Enables the `wgpu_cooperative_matrix` extension, native only.
    WgpuCooperativeMatrix,
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
    /// Enables the `@builtin(primitive_index)` attribute in WGSL.
    ///
    /// In the WGSL standard, this corresponds to [`enable primitive-index;`].
    ///
    /// [`enable primitive-index;`]: https://www.w3.org/TR/WGSL/#extension-primitive_index
    PrimitiveIndex,
}

impl UnimplementedEnableExtension {
    pub(crate) const fn tracking_issue_num(self) -> u16 {
        match self {
            Self::Subgroups => 5555,
            Self::PrimitiveIndex => 8236,
        }
    }
}

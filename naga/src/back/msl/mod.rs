/*!
Backend for [MSL][msl] (Metal Shading Language).

This backend does not support the [`SHADER_INT64_ATOMIC_ALL_OPS`][all-atom]
capability.

## Binding model

Metal's bindings are flat per resource. Since there isn't an obvious mapping
from SPIR-V's descriptor sets, we require a separate mapping provided in the options.
This mapping may have one or more resource end points for each descriptor set + index
pair.

## Entry points

Even though MSL and our IR appear to be similar in that the entry points in both can
accept arguments and return values, the restrictions are different.
MSL allows the varyings to be either in separate arguments, or inside a single
`[[stage_in]]` struct. We gather input varyings and form this artificial structure.
We also add all the (non-Private) globals into the arguments.

At the beginning of the entry point, we assign the local constants and re-compose
the arguments as they are declared on IR side, so that the rest of the logic can
pretend that MSL doesn't have all the restrictions it has.

For the result type, if it's a structure, we re-compose it with a temporary value
holding the result.

[msl]: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
[all-atom]: crate::valid::Capabilities::SHADER_INT64_ATOMIC_ALL_OPS

## Pointer-typed bounds-checked expressions and OOB locals

MSL (unlike HLSL and GLSL) has native support for pointer-typed function
arguments. When the [`BoundsCheckPolicy`] is `ReadZeroSkipWrite` and an
out-of-bounds index expression is used for such an argument, our strategy is to
pass a pointer to a dummy variable. These dummy variables are called "OOB
locals". We emit at most one OOB local per function for each type, since all
expressions producing a result of that type can share the same OOB local. (Note
that the OOB local mechanism is not actually implementing "skip write", nor even
"read zero" in some cases of read-after-write, but doing so would require
additional effort and the difference is unlikely to matter.)

[`BoundsCheckPolicy`]: crate::proc::BoundsCheckPolicy

## External textures

Support for [`crate::ImageClass::External`] textures is implemented by lowering
each external texture global variable to 3 `texture2d<float, sample>`s, and a
constant buffer of type `NagaExternalTextureParams`. This provides up to 3
planes of texture data (for example single planar RGBA, or separate Y, Cb, and
Cr planes), and the parameters buffer containing information describing how to
handle these correctly. The bind target to use for each of these globals is
specified via the [`BindTarget::external_texture`] field of the relevant
entries in [`EntryPointResources::resources`].

External textures are supported by WGSL's `textureDimensions()`,
`textureLoad()`, and `textureSampleBaseClampToEdge()` built-in functions. These
are implemented using helper functions. See the following functions for how
these are generated:
 * `Writer::write_wrapped_image_query`
 * `Writer::write_wrapped_image_load`
 * `Writer::write_wrapped_image_sample`

The lowered global variables for each external texture global are passed to the
entry point as separate arguments (see "Entry points" above). However, they are
then wrapped in a struct to allow them to be conveniently passed to user
defined and helper functions. See `writer::EXTERNAL_TEXTURE_WRAPPER_STRUCT`.
*/

use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use core::fmt::{Error as FmtError, Write};

use crate::{arena::Handle, ir, proc::index, valid::ModuleInfo};

mod keywords;
pub mod sampler;
mod writer;

pub use writer::Writer;
pub use wst::msl::*;

enum ResolvedBinding {
    BuiltIn(crate::BuiltIn),
    Attribute(u32),
    Color {
        location: u32,
        blend_src: Option<u32>,
    },
    User {
        prefix: &'static str,
        index: u32,
        interpolation: Option<ResolvedInterpolation>,
    },
    Resource(BindTarget),
}

#[derive(Copy, Clone)]
enum ResolvedInterpolation {
    CenterPerspective,
    CenterNoPerspective,
    CentroidPerspective,
    CentroidNoPerspective,
    SamplePerspective,
    SampleNoPerspective,
    Flat,
}

// Note: some of these should be removed in favor of proper IR validation.

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Format(#[from] FmtError),
    #[error("bind target {0:?} is empty")]
    UnimplementedBindTarget(BindTarget),
    #[error("composing of {0:?} is not implemented yet")]
    UnsupportedCompose(Handle<crate::Type>),
    #[error("operation {0:?} is not implemented yet")]
    UnsupportedBinaryOp(crate::BinaryOperator),
    #[error("standard function '{0}' is not implemented yet")]
    UnsupportedCall(String),
    #[error("feature '{0}' is not implemented yet")]
    FeatureNotImplemented(String),
    #[error("internal naga error: module should not have validated: {0}")]
    GenericValidation(String),
    #[error("BuiltIn {0:?} is not supported")]
    UnsupportedBuiltIn(crate::BuiltIn),
    #[error("capability {0:?} is not supported")]
    CapabilityNotSupported(crate::valid::Capabilities),
    #[error("attribute '{0}' is not supported for target MSL version")]
    UnsupportedAttribute(String),
    #[error("function '{0}' is not supported for target MSL version")]
    UnsupportedFunction(String),
    #[error("can not use writeable storage buffers in fragment stage prior to MSL 1.2")]
    UnsupportedWriteableStorageBuffer,
    #[error("can not use writeable storage textures in {0:?} stage prior to MSL 1.2")]
    UnsupportedWriteableStorageTexture(ir::ShaderStage),
    #[error("can not use read-write storage textures prior to MSL 1.2")]
    UnsupportedRWStorageTexture,
    #[error("array of '{0}' is not supported for target MSL version")]
    UnsupportedArrayOf(String),
    #[error("array of type '{0:?}' is not supported")]
    UnsupportedArrayOfType(Handle<crate::Type>),
    #[error("ray tracing is not supported prior to MSL 2.4")]
    UnsupportedRayTracing,
    #[error("cooperative matrix is not supported prior to MSL 2.3")]
    UnsupportedCooperativeMatrix,
    #[error("overrides should not be present at this stage")]
    Override,
    #[error("bitcasting to {0:?} is not supported")]
    UnsupportedBitCast(crate::TypeInner),
    #[error(transparent)]
    ResolveArraySizeError(#[from] crate::proc::ResolveArraySizeError),
    #[error("entry point with stage {0:?} and name '{1}' not found")]
    EntryPointNotFound(ir::ShaderStage, String),
}

/// Points in the MSL code where we might emit a pipeline input or output.
///
/// Note that, even though vertex shaders' outputs are always fragment
/// shaders' inputs, we still need to distinguish `VertexOutput` and
/// `FragmentInput`, since there are certain differences in the way
/// [`ResolvedBinding`s] are represented on either side.
///
/// [`ResolvedBinding`s]: ResolvedBinding
#[derive(Clone, Copy, Debug)]
enum LocationMode {
    /// Input to the vertex shader.
    VertexInput,

    /// Output from the vertex shader.
    VertexOutput,

    /// Input to the fragment shader.
    FragmentInput,

    /// Output from the fragment shader.
    FragmentOutput,

    /// Compute shader input or output.
    Uniform,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct Options {
    /// (Major, Minor) target version of the Metal Shading Language.
    pub lang_version: (u8, u8),
    /// Map of entry-point resources, indexed by entry point function name, to slots.
    pub per_entry_point_map: EntryPointResourceMap,
    /// Samplers to be inlined into the code.
    pub inline_samplers: Vec<sampler::InlineSampler>,
    /// Make it possible to link different stages via SPIRV-Cross.
    pub spirv_cross_compatibility: bool,
    /// Don't panic on missing bindings, instead generate invalid MSL.
    pub fake_missing_bindings: bool,
    /// Bounds checking policies.
    pub bounds_check_policies: index::BoundsCheckPolicies,
    /// Should workgroup variables be zero initialized (by polyfilling)?
    pub zero_initialize_workgroup_memory: bool,
    /// If set, loops will have code injected into them, forcing the compiler
    /// to think the number of iterations is bounded.
    pub force_loop_bounding: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            lang_version: (1, 0),
            per_entry_point_map: EntryPointResourceMap::default(),
            inline_samplers: Vec::new(),
            spirv_cross_compatibility: false,
            fake_missing_bindings: true,
            bounds_check_policies: index::BoundsCheckPolicies::default(),
            zero_initialize_workgroup_memory: true,
            force_loop_bounding: true,
        }
    }
}

/// A subset of options that are meant to be changed per pipeline.
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
#[cfg_attr(feature = "deserialize", serde(default))]
pub struct PipelineOptions {
    /// The entry point to write.
    ///
    /// Entry points are identified by a shader stage specification,
    /// and a name.
    ///
    /// If `None`, all entry points will be written. If `Some` and the entry
    /// point is not found, an error will be thrown while writing.
    ///
    /// The primitive topology is only used checked against point topology for vertex shaders.
    pub entry_point: Option<(ir::ShaderStage, String, wst::PrimitiveTopology)>,

    /// If set, when generating the Metal vertex shader, transform it
    /// to receive the vertex buffers, lengths, and vertex id as args,
    /// and bounds-check the vertex id and use the index into the
    /// vertex buffers to access attributes, rather than using Metal's
    /// [[stage-in]] assembled attribute data. This is true by default,
    /// but remains configurable for use by tests via deserialization
    /// of this struct. There is no user-facing way to set this value.
    pub vertex_pulling_transform: bool,

    /// vertex_buffer_mappings are used during shader translation to
    /// support vertex pulling.
    pub vertex_buffer_mappings: Vec<VertexBufferMapping>,
}

impl Options {
    fn resolve_local_binding(
        &self,
        binding: &crate::Binding,
        mode: LocationMode,
    ) -> Result<ResolvedBinding, Error> {
        match *binding {
            crate::Binding::BuiltIn(mut built_in) => {
                match built_in {
                    crate::BuiltIn::Position { ref mut invariant } => {
                        if *invariant && self.lang_version < (2, 1) {
                            return Err(Error::UnsupportedAttribute("invariant".to_string()));
                        }

                        // The 'invariant' attribute may only appear on vertex
                        // shader outputs, not fragment shader inputs.
                        if !matches!(mode, LocationMode::VertexOutput) {
                            *invariant = false;
                        }
                    }
                    crate::BuiltIn::BaseInstance if self.lang_version < (1, 2) => {
                        return Err(Error::UnsupportedAttribute("base_instance".to_string()));
                    }
                    crate::BuiltIn::InstanceIndex if self.lang_version < (1, 2) => {
                        return Err(Error::UnsupportedAttribute("instance_id".to_string()));
                    }
                    // macOS: Since Metal 2.2
                    // iOS: Since Metal 2.3 (check depends on https://github.com/gfx-rs/wgpu/issues/4414)
                    crate::BuiltIn::PrimitiveIndex if self.lang_version < (2, 3) => {
                        return Err(Error::UnsupportedAttribute("primitive_id".to_string()));
                    }
                    // macOS: since Metal 2.3
                    // iOS: Since Metal 2.2
                    // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf#page=114
                    crate::BuiltIn::ViewIndex if self.lang_version < (2, 2) => {
                        return Err(Error::UnsupportedAttribute("amplification_id".to_string()));
                    }
                    // macOS: Since Metal 2.2
                    // iOS: Since Metal 2.3 (check depends on https://github.com/gfx-rs/wgpu/issues/4414)
                    crate::BuiltIn::Barycentric { .. } if self.lang_version < (2, 3) => {
                        return Err(Error::UnsupportedAttribute("barycentric_coord".to_string()));
                    }
                    _ => {}
                }

                Ok(ResolvedBinding::BuiltIn(built_in))
            }
            crate::Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive: _,
            } => match mode {
                LocationMode::VertexInput => Ok(ResolvedBinding::Attribute(location)),
                LocationMode::FragmentOutput => {
                    if blend_src.is_some() && self.lang_version < (1, 2) {
                        return Err(Error::UnsupportedAttribute("blend_src".to_string()));
                    }
                    Ok(ResolvedBinding::Color {
                        location,
                        blend_src,
                    })
                }
                LocationMode::VertexOutput | LocationMode::FragmentInput => {
                    Ok(ResolvedBinding::User {
                        prefix: if self.spirv_cross_compatibility {
                            "locn"
                        } else {
                            "loc"
                        },
                        index: location,
                        interpolation: {
                            // unwrap: The verifier ensures that vertex shader outputs and fragment
                            // shader inputs always have fully specified interpolation, and that
                            // sampling is `None` only for Flat interpolation.
                            let interpolation = interpolation.unwrap();
                            let sampling = sampling.unwrap_or(crate::Sampling::Center);
                            Some(ResolvedInterpolation::from_binding(interpolation, sampling))
                        },
                    })
                }
                LocationMode::Uniform => Err(Error::GenericValidation(format!(
                    "Unexpected Binding::Location({location}) for the Uniform mode"
                ))),
            },
        }
    }

    fn get_entry_point_resources(&self, ep: &crate::EntryPoint) -> Option<&EntryPointResources> {
        self.per_entry_point_map.get(&ep.name)
    }

    fn get_resource_binding_target(
        &self,
        ep: &crate::EntryPoint,
        res_binding: &crate::ResourceBinding,
    ) -> Option<&BindTarget> {
        self.get_entry_point_resources(ep)
            .and_then(|res| res.resources.get(res_binding))
    }

    fn resolve_resource_binding(
        &self,
        ep: &crate::EntryPoint,
        res_binding: &crate::ResourceBinding,
    ) -> Result<ResolvedBinding, EntryPointError> {
        let target = self.get_resource_binding_target(ep, res_binding);
        match target {
            Some(target) => Ok(ResolvedBinding::Resource(target.clone())),
            None if self.fake_missing_bindings => Ok(ResolvedBinding::User {
                prefix: "fake",
                index: 0,
                interpolation: None,
            }),
            None => Err(EntryPointError::MissingBindTarget(*res_binding)),
        }
    }

    fn resolve_immediates(
        &self,
        ep: &crate::EntryPoint,
    ) -> Result<ResolvedBinding, EntryPointError> {
        let slot = self
            .get_entry_point_resources(ep)
            .and_then(|res| res.immediates_buffer);
        match slot {
            Some(slot) => Ok(ResolvedBinding::Resource(BindTarget {
                buffer: Some(slot),
                ..Default::default()
            })),
            None if self.fake_missing_bindings => Ok(ResolvedBinding::User {
                prefix: "fake",
                index: 0,
                interpolation: None,
            }),
            None => Err(EntryPointError::MissingImmediateData),
        }
    }

    fn resolve_sizes_buffer(
        &self,
        ep: &crate::EntryPoint,
    ) -> Result<ResolvedBinding, EntryPointError> {
        let slot = self
            .get_entry_point_resources(ep)
            .and_then(|res| res.sizes_buffer);
        match slot {
            Some(slot) => Ok(ResolvedBinding::Resource(BindTarget {
                buffer: Some(slot),
                ..Default::default()
            })),
            None if self.fake_missing_bindings => Ok(ResolvedBinding::User {
                prefix: "fake",
                index: 0,
                interpolation: None,
            }),
            None => Err(EntryPointError::MissingSizesBuffer),
        }
    }
}

impl ResolvedBinding {
    fn as_inline_sampler<'a>(&self, options: &'a Options) -> Option<&'a sampler::InlineSampler> {
        match *self {
            Self::Resource(BindTarget {
                sampler: Some(BindSamplerTarget::Inline(index)),
                ..
            }) => Some(&options.inline_samplers[index as usize]),
            _ => None,
        }
    }

    fn try_fmt<W: Write>(&self, out: &mut W) -> Result<(), Error> {
        write!(out, " [[")?;
        match *self {
            Self::BuiltIn(built_in) => {
                use crate::BuiltIn as Bi;
                let name = match built_in {
                    Bi::Position { invariant: false } => "position",
                    Bi::Position { invariant: true } => "position, invariant",
                    Bi::ViewIndex => "amplification_id",
                    // vertex
                    Bi::BaseInstance => "base_instance",
                    Bi::BaseVertex => "base_vertex",
                    Bi::ClipDistance => "clip_distance",
                    Bi::InstanceIndex => "instance_id",
                    Bi::PointSize => "point_size",
                    Bi::VertexIndex => "vertex_id",
                    // fragment
                    Bi::FragDepth => "depth(any)",
                    Bi::PointCoord => "point_coord",
                    Bi::FrontFacing => "front_facing",
                    Bi::PrimitiveIndex => "primitive_id",
                    Bi::Barycentric { perspective: true } => "barycentric_coord",
                    Bi::Barycentric { perspective: false } => {
                        "barycentric_coord, center_no_perspective"
                    }
                    Bi::SampleIndex => "sample_id",
                    Bi::SampleMask => "sample_mask",
                    // compute
                    Bi::GlobalInvocationId => "thread_position_in_grid",
                    Bi::LocalInvocationId => "thread_position_in_threadgroup",
                    Bi::LocalInvocationIndex => "thread_index_in_threadgroup",
                    Bi::WorkGroupId => "threadgroup_position_in_grid",
                    Bi::WorkGroupSize => "dispatch_threads_per_threadgroup",
                    Bi::NumWorkGroups => "threadgroups_per_grid",
                    // subgroup
                    Bi::NumSubgroups => "simdgroups_per_threadgroup",
                    Bi::SubgroupId => "simdgroup_index_in_threadgroup",
                    Bi::SubgroupSize => "threads_per_simdgroup",
                    Bi::SubgroupInvocationId => "thread_index_in_simdgroup",
                    Bi::CullDistance | Bi::DrawIndex => {
                        return Err(Error::UnsupportedBuiltIn(built_in))
                    }
                    Bi::CullPrimitive => "primitive_culled",
                    // TODO: figure out how to make this written as a function call
                    Bi::PointIndex | Bi::LineIndices | Bi::TriangleIndices => unimplemented!(),
                    Bi::MeshTaskSize
                    | Bi::VertexCount
                    | Bi::PrimitiveCount
                    | Bi::Vertices
                    | Bi::Primitives
                    | Bi::RayInvocationId
                    | Bi::NumRayInvocations
                    | Bi::InstanceCustomData
                    | Bi::GeometryIndex
                    | Bi::WorldRayOrigin
                    | Bi::WorldRayDirection
                    | Bi::ObjectRayOrigin
                    | Bi::ObjectRayDirection
                    | Bi::RayTmin
                    | Bi::RayTCurrentMax
                    | Bi::ObjectToWorld
                    | Bi::WorldToObject
                    | Bi::HitKind => unreachable!(),
                };
                write!(out, "{name}")?;
            }
            Self::Attribute(index) => write!(out, "attribute({index})")?,
            Self::Color {
                location,
                blend_src,
            } => {
                if let Some(blend_src) = blend_src {
                    write!(out, "color({location}) index({blend_src})")?
                } else {
                    write!(out, "color({location})")?
                }
            }
            Self::User {
                prefix,
                index,
                interpolation,
            } => {
                write!(out, "user({prefix}{index})")?;
                if let Some(interpolation) = interpolation {
                    write!(out, ", ")?;
                    interpolation.try_fmt(out)?;
                }
            }
            Self::Resource(ref target) => {
                if let Some(id) = target.buffer {
                    write!(out, "buffer({id})")?;
                } else if let Some(id) = target.texture {
                    write!(out, "texture({id})")?;
                } else if let Some(BindSamplerTarget::Resource(id)) = target.sampler {
                    write!(out, "sampler({id})")?;
                } else {
                    return Err(Error::UnimplementedBindTarget(target.clone()));
                }
            }
        }
        write!(out, "]]")?;
        Ok(())
    }
}

impl ResolvedInterpolation {
    const fn from_binding(interpolation: crate::Interpolation, sampling: crate::Sampling) -> Self {
        use crate::Interpolation as I;
        use crate::Sampling as S;

        match (interpolation, sampling) {
            (I::Perspective, S::Center) => Self::CenterPerspective,
            (I::Perspective, S::Centroid) => Self::CentroidPerspective,
            (I::Perspective, S::Sample) => Self::SamplePerspective,
            (I::Linear, S::Center) => Self::CenterNoPerspective,
            (I::Linear, S::Centroid) => Self::CentroidNoPerspective,
            (I::Linear, S::Sample) => Self::SampleNoPerspective,
            (I::Flat, _) => Self::Flat,
            _ => unreachable!(),
        }
    }

    fn try_fmt<W: Write>(self, out: &mut W) -> Result<(), Error> {
        let identifier = match self {
            Self::CenterPerspective => "center_perspective",
            Self::CenterNoPerspective => "center_no_perspective",
            Self::CentroidPerspective => "centroid_perspective",
            Self::CentroidNoPerspective => "centroid_no_perspective",
            Self::SamplePerspective => "sample_perspective",
            Self::SampleNoPerspective => "sample_no_perspective",
            Self::Flat => "flat",
        };
        out.write_str(identifier)?;
        Ok(())
    }
}

pub fn write_string(
    module: &crate::Module,
    info: &ModuleInfo,
    options: &Options,
    pipeline_options: &PipelineOptions,
) -> Result<(String, TranslationInfo), Error> {
    let mut w = Writer::new(String::new());
    let info = w.write(module, info, options, pipeline_options)?;
    Ok((w.finish(), info))
}

#[test]
fn test_error_size() {
    assert_eq!(size_of::<Error>(), 40);
}

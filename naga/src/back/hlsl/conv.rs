use std::borrow::Cow;

use crate::proc::Alignment;

use super::Error;

impl crate::ScalarKind {
    pub(super) fn to_hlsl_cast(self) -> &'static str {
        match self {
            Self::Float => "asfloat",
            Self::Sint => "asint",
            Self::Uint => "asuint",
            Self::Bool => unreachable!(),
        }
    }

    /// Helper function that returns scalar related strings
    ///
    /// <https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-scalar>
    pub(super) const fn to_hlsl_str(self, width: crate::Bytes) -> Result<&'static str, Error> {
        match self {
            Self::Sint => Ok("int"),
            Self::Uint => Ok("uint"),
            Self::Float => match width {
                2 => Ok("half"),
                4 => Ok("float"),
                8 => Ok("double"),
                _ => Err(Error::UnsupportedScalar(self, width)),
            },
            Self::Bool => Ok("bool"),
        }
    }
}

impl crate::TypeInner {
    pub(super) const fn is_matrix(&self) -> bool {
        match *self {
            Self::Matrix { .. } => true,
            _ => false,
        }
    }

    pub(super) fn size_hlsl(&self, gctx: crate::proc::GlobalCtx) -> u32 {
        match *self {
            Self::Matrix {
                columns,
                rows,
                width,
            } => {
                let stride = Alignment::from(rows) * width as u32;
                let last_row_size = rows as u32 * width as u32;
                ((columns as u32 - 1) * stride) + last_row_size
            }
            Self::Array { base, size, stride } => {
                let count = match size {
                    crate::ArraySize::Constant(size) => size.get(),
                    // A dynamically-sized array has to have at least one element
                    crate::ArraySize::Dynamic => 1,
                };
                let last_el_size = gctx.types[base].inner.size_hlsl(gctx);
                ((count - 1) * stride) + last_el_size
            }
            _ => self.size(gctx),
        }
    }

    /// Used to generate the name of the wrapped type constructor
    pub(super) fn hlsl_type_id<'a>(
        base: crate::Handle<crate::Type>,
        gctx: crate::proc::GlobalCtx,
        names: &'a crate::FastHashMap<crate::proc::NameKey, String>,
    ) -> Result<Cow<'a, str>, Error> {
        Ok(match gctx.types[base].inner {
            crate::TypeInner::Scalar { kind, width } => Cow::Borrowed(kind.to_hlsl_str(width)?),
            crate::TypeInner::Vector { size, kind, width } => Cow::Owned(format!(
                "{}{}",
                kind.to_hlsl_str(width)?,
                crate::back::vector_size_str(size)
            )),
            crate::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => Cow::Owned(format!(
                "{}{}x{}",
                crate::ScalarKind::Float.to_hlsl_str(width)?,
                crate::back::vector_size_str(columns),
                crate::back::vector_size_str(rows),
            )),
            crate::TypeInner::Array {
                base,
                size: crate::ArraySize::Constant(size),
                ..
            } => Cow::Owned(format!(
                "array{size}_{}_",
                Self::hlsl_type_id(base, gctx, names)?
            )),
            crate::TypeInner::Struct { .. } => {
                Cow::Borrowed(&names[&crate::proc::NameKey::Type(base)])
            }
            _ => unreachable!(),
        })
    }
}

impl crate::StorageFormat {
    pub(super) const fn to_hlsl_str(self) -> &'static str {
        match self {
            Self::R16Float => "float",
            Self::R8Unorm | Self::R16Unorm => "unorm float",
            Self::R8Snorm | Self::R16Snorm => "snorm float",
            Self::R8Uint | Self::R16Uint => "uint",
            Self::R8Sint | Self::R16Sint => "int",

            Self::Rg16Float => "float2",
            Self::Rg8Unorm | Self::Rg16Unorm => "unorm float2",
            Self::Rg8Snorm | Self::Rg16Snorm => "snorm float2",

            Self::Rg8Sint | Self::Rg16Sint => "int2",
            Self::Rg8Uint | Self::Rg16Uint => "uint2",

            Self::Rg11b10Float => "float3",

            Self::Rgba16Float | Self::R32Float | Self::Rg32Float | Self::Rgba32Float => "float4",
            Self::Rgba8Unorm | Self::Bgra8Unorm | Self::Rgba16Unorm | Self::Rgb10a2Unorm => {
                "unorm float4"
            }
            Self::Rgba8Snorm | Self::Rgba16Snorm => "snorm float4",

            Self::Rgba8Uint
            | Self::Rgba16Uint
            | Self::R32Uint
            | Self::Rg32Uint
            | Self::Rgba32Uint
            | Self::Rgb10a2Uint => "uint4",
            Self::Rgba8Sint
            | Self::Rgba16Sint
            | Self::R32Sint
            | Self::Rg32Sint
            | Self::Rgba32Sint => "int4",
        }
    }
}

impl crate::BuiltIn {
    pub(super) fn to_hlsl_str(self) -> Result<&'static str, Error> {
        Ok(match self {
            Self::Position { .. } => "SV_Position",
            // vertex
            Self::ClipDistance => "SV_ClipDistance",
            Self::CullDistance => "SV_CullDistance",
            Self::InstanceIndex => "SV_InstanceID",
            Self::VertexIndex => "SV_VertexID",
            // fragment
            Self::FragDepth => "SV_Depth",
            Self::FrontFacing => "SV_IsFrontFace",
            Self::PrimitiveIndex => "SV_PrimitiveID",
            Self::SampleIndex => "SV_SampleIndex",
            Self::SampleMask => "SV_Coverage",
            // compute
            Self::GlobalInvocationId => "SV_DispatchThreadID",
            Self::LocalInvocationId => "SV_GroupThreadID",
            Self::LocalInvocationIndex => "SV_GroupIndex",
            Self::WorkGroupId => "SV_GroupID",
            // The specific semantic we use here doesn't matter, because references
            // to this field will get replaced with references to `SPECIAL_CBUF_VAR`
            // in `Writer::write_expr`.
            Self::NumWorkGroups => "SV_GroupID",

            Self::NumSubgroups | Self::SubgroupId => todo!(),
            Self::SubgroupInvocationId
            | Self::SubgroupSize
            | Self::BaseInstance
            | Self::BaseVertex
            | Self::WorkGroupSize => return Err(Error::Unimplemented(format!("builtin {self:?}"))),
            Self::PointSize | Self::ViewIndex | Self::PointCoord => {
                return Err(Error::Custom(format!("Unsupported builtin {self:?}")))
            }
        })
    }
}

impl crate::Interpolation {
    /// Return the string corresponding to the HLSL interpolation qualifier.
    pub(super) const fn to_hlsl_str(self) -> Option<&'static str> {
        match self {
            // Would be "linear", but it's the default interpolation in SM4 and up
            // https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/dx-graphics-hlsl-struct#interpolation-modifiers-introduced-in-shader-model-4
            Self::Perspective => None,
            Self::Linear => Some("noperspective"),
            Self::Flat => Some("nointerpolation"),
        }
    }
}

impl crate::Sampling {
    /// Return the HLSL auxiliary qualifier for the given sampling value.
    pub(super) const fn to_hlsl_str(self) -> Option<&'static str> {
        match self {
            Self::Center => None,
            Self::Centroid => Some("centroid"),
            Self::Sample => Some("sample"),
        }
    }
}

impl crate::AtomicFunction {
    /// Return the HLSL suffix for the `InterlockedXxx` method.
    pub(super) const fn to_hlsl_suffix(self) -> &'static str {
        match self {
            Self::Add | Self::Subtract => "Add",
            Self::And => "And",
            Self::InclusiveOr => "Or",
            Self::ExclusiveOr => "Xor",
            Self::Min => "Min",
            Self::Max => "Max",
            Self::Exchange { compare: None } => "Exchange",
            Self::Exchange { .. } => "", //TODO
        }
    }
}

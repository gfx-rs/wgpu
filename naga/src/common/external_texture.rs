/// A member of the `ExternalTextureParams` struct.
///
/// The numeric values of this enum are used to generate struct access operations in the
/// SPIR-V backend. The order must match the other definitions of the struct. Consistency
/// with Naga's special type definition is checked by a debug assertion in
/// [`crate::Module::generate_external_texture_types`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[cfg_attr(test, derive(strum::EnumCount))]
#[repr(u32)]
pub(crate) enum ExternalTextureParameter {
    YuvConversionMatrix = 0,
    GamutConversionMatrix,
    SrcTf,
    DstTf,
    SampleTransform,
    LoadTransform,
    Size,
    NumPlanes,
}

impl ExternalTextureParameter {
    pub const fn name(&self) -> &'static str {
        use ExternalTextureParameter as P;
        match *self {
            P::YuvConversionMatrix => "yuv_conversion_matrix",
            P::GamutConversionMatrix => "gamut_conversion_matrix",
            P::SrcTf => "src_tf",
            P::DstTf => "dst_tf",
            P::SampleTransform => "sample_transform",
            P::LoadTransform => "load_transform",
            P::Size => "size",
            P::NumPlanes => "num_planes",
        }
    }
}

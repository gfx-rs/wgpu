impl super::PrivateCapability {
    pub(super) fn describe_texture_format(
        &self,
        format: wgt::TextureFormat,
    ) -> super::FormatDescription {
        use super::VertexAttribKind as Vak;
        use wgt::TextureFormat as Tf;

        let (tex_internal, tex_external, data_type, num_components, va_kind) = match format {
            Tf::R8Unorm => (glow::R8, glow::RED, glow::UNSIGNED_BYTE, 1, Vak::Float),
            Tf::R8Snorm => (glow::R8, glow::RED, glow::BYTE, 1, Vak::Float),
            Tf::R8Uint => (
                glow::R8UI,
                glow::RED_INTEGER,
                glow::UNSIGNED_BYTE,
                1,
                Vak::Integer,
            ),
            Tf::R8Sint => (glow::R8I, glow::RED_INTEGER, glow::BYTE, 1, Vak::Integer),
            Tf::R16Uint => (
                glow::R16UI,
                glow::RED_INTEGER,
                glow::UNSIGNED_SHORT,
                1,
                Vak::Integer,
            ),
            Tf::R16Sint => (glow::R16I, glow::RED_INTEGER, glow::SHORT, 1, Vak::Integer),
            Tf::R16Float => (glow::R16F, glow::RED, glow::UNSIGNED_SHORT, 1, Vak::Float),
            Tf::Rg8Unorm => (glow::RG8, glow::RG, glow::UNSIGNED_BYTE, 2, Vak::Float),
            Tf::Rg8Snorm => (glow::RG8, glow::RG, glow::BYTE, 2, Vak::Float),
            Tf::Rg8Uint => (
                glow::RG8UI,
                glow::RG_INTEGER,
                glow::UNSIGNED_BYTE,
                2,
                Vak::Integer,
            ),
            Tf::Rg8Sint => (glow::RG8I, glow::RG_INTEGER, glow::BYTE, 2, Vak::Integer),
            Tf::R32Uint => (
                glow::R32UI,
                glow::RED_INTEGER,
                glow::UNSIGNED_INT,
                1,
                Vak::Integer,
            ),
            Tf::R32Sint => (glow::R32I, glow::RED_INTEGER, glow::INT, 1, Vak::Integer),
            Tf::R32Float => (glow::R32F, glow::RED, glow::FLOAT, 1, Vak::Float),
            Tf::Rg16Uint => (
                glow::RG16UI,
                glow::RG_INTEGER,
                glow::UNSIGNED_SHORT,
                2,
                Vak::Integer,
            ),
            Tf::Rg16Sint => (glow::RG16I, glow::RG_INTEGER, glow::SHORT, 2, Vak::Integer),
            Tf::Rg16Float => (glow::RG16F, glow::RG, glow::UNSIGNED_SHORT, 2, Vak::Float),
            Tf::Rgba8Unorm => (glow::RGBA8, glow::RGBA, glow::UNSIGNED_BYTE, 4, Vak::Float),
            Tf::Rgba8UnormSrgb => (
                glow::SRGB8_ALPHA8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                4,
                Vak::Float,
            ),
            Tf::Bgra8UnormSrgb => (
                glow::SRGB8_ALPHA8,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                4,
                Vak::Float,
            ), //TODO?
            Tf::Rgba8Snorm => (glow::RGBA8, glow::RGBA, glow::BYTE, 4, Vak::Float),
            Tf::Bgra8Unorm => (glow::RGBA8, glow::BGRA, glow::UNSIGNED_BYTE, 4, Vak::Float),
            Tf::Rgba8Uint => (
                glow::RGBA8UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_BYTE,
                4,
                Vak::Integer,
            ),
            Tf::Rgba8Sint => (
                glow::RGBA8I,
                glow::RGBA_INTEGER,
                glow::BYTE,
                4,
                Vak::Integer,
            ),
            Tf::Rgb10a2Unorm => (
                glow::RGB10_A2,
                glow::RGBA,
                glow::UNSIGNED_INT_2_10_10_10_REV,
                1,
                Vak::Integer,
            ),
            Tf::Rg11b10Float => (
                glow::R11F_G11F_B10F,
                glow::RGB,
                glow::UNSIGNED_INT_10F_11F_11F_REV,
                1,
                Vak::Integer,
            ),
            Tf::Rg32Uint => (
                glow::RG32UI,
                glow::RG_INTEGER,
                glow::UNSIGNED_INT,
                2,
                Vak::Integer,
            ),
            Tf::Rg32Sint => (glow::RG32I, glow::RG_INTEGER, glow::INT, 2, Vak::Integer),
            Tf::Rg32Float => (glow::RG32F, glow::RG, glow::FLOAT, 2, Vak::Float),
            Tf::Rgba16Uint => (
                glow::RGBA16UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_SHORT,
                4,
                Vak::Integer,
            ),
            Tf::Rgba16Sint => (
                glow::RGBA16I,
                glow::RGBA_INTEGER,
                glow::SHORT,
                4,
                Vak::Integer,
            ),
            Tf::Rgba16Float => (glow::RGBA16F, glow::RG, glow::UNSIGNED_SHORT, 4, Vak::Float),
            Tf::Rgba32Uint => (
                glow::RGBA32UI,
                glow::RGBA_INTEGER,
                glow::UNSIGNED_INT,
                4,
                Vak::Integer,
            ),
            Tf::Rgba32Sint => (
                glow::RGBA32I,
                glow::RGBA_INTEGER,
                glow::INT,
                4,
                Vak::Integer,
            ),
            Tf::Rgba32Float => (glow::RGBA32F, glow::RGBA, glow::FLOAT, 4, Vak::Float),
            Tf::Depth32Float => (
                glow::DEPTH_COMPONENT32F,
                glow::DEPTH_COMPONENT,
                glow::FLOAT,
                1,
                Vak::Float,
            ),
            Tf::Depth24Plus => (
                glow::DEPTH_COMPONENT24,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_NORMALIZED,
                2,
                Vak::Float,
            ),
            Tf::Depth24PlusStencil8 => (
                glow::DEPTH24_STENCIL8,
                glow::DEPTH_COMPONENT,
                glow::UNSIGNED_INT,
                2,
                Vak::Float,
            ),
            Tf::Bc1RgbaUnorm
            | Tf::Bc1RgbaUnormSrgb
            | Tf::Bc2RgbaUnorm
            | Tf::Bc2RgbaUnormSrgb
            | Tf::Bc3RgbaUnorm
            | Tf::Bc3RgbaUnormSrgb
            | Tf::Bc4RUnorm
            | Tf::Bc4RSnorm
            | Tf::Bc5RgUnorm
            | Tf::Bc5RgSnorm
            | Tf::Bc6hRgbSfloat
            | Tf::Bc6hRgbUfloat
            | Tf::Bc7RgbaUnorm
            | Tf::Bc7RgbaUnormSrgb
            | Tf::Etc2RgbUnorm
            | Tf::Etc2RgbUnormSrgb
            | Tf::Etc2RgbA1Unorm
            | Tf::Etc2RgbA1UnormSrgb
            | Tf::EacRUnorm
            | Tf::EacRSnorm
            | Tf::EtcRgUnorm
            | Tf::EtcRgSnorm
            | Tf::Astc4x4RgbaUnorm
            | Tf::Astc4x4RgbaUnormSrgb
            | Tf::Astc5x4RgbaUnorm
            | Tf::Astc5x4RgbaUnormSrgb
            | Tf::Astc5x5RgbaUnorm
            | Tf::Astc5x5RgbaUnormSrgb
            | Tf::Astc6x5RgbaUnorm
            | Tf::Astc6x5RgbaUnormSrgb
            | Tf::Astc6x6RgbaUnorm
            | Tf::Astc6x6RgbaUnormSrgb
            | Tf::Astc8x5RgbaUnorm
            | Tf::Astc8x5RgbaUnormSrgb
            | Tf::Astc8x6RgbaUnorm
            | Tf::Astc8x6RgbaUnormSrgb
            | Tf::Astc10x5RgbaUnorm
            | Tf::Astc10x5RgbaUnormSrgb
            | Tf::Astc10x6RgbaUnorm
            | Tf::Astc10x6RgbaUnormSrgb
            | Tf::Astc8x8RgbaUnorm
            | Tf::Astc8x8RgbaUnormSrgb
            | Tf::Astc10x8RgbaUnorm
            | Tf::Astc10x8RgbaUnormSrgb
            | Tf::Astc10x10RgbaUnorm
            | Tf::Astc10x10RgbaUnormSrgb
            | Tf::Astc12x10RgbaUnorm
            | Tf::Astc12x10RgbaUnormSrgb
            | Tf::Astc12x12RgbaUnorm
            | Tf::Astc12x12RgbaUnormSrgb => unimplemented!(),
        };

        super::FormatDescription {
            tex_internal,
            tex_external,
            data_type,
            num_components,
            va_kind,
        }
    }
}

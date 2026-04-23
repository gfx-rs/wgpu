use core::{cmp::Ordering, fmt};

// Must match code in glsl_built_in
pub const FIRST_INSTANCE_BINDING: &str = "naga_vs_first_instance";

/// List of supported `core` GLSL versions.
pub const SUPPORTED_CORE_VERSIONS: &[u16] = &[140, 150, 330, 400, 410, 420, 430, 440, 450, 460];
/// List of supported `es` GLSL versions.
pub const SUPPORTED_ES_VERSIONS: &[u16] = &[300, 310, 320];

/// A GLSL version.
#[derive(Debug, Copy, Clone, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum Version {
    /// `core` GLSL.
    Desktop(u16),
    /// `es` GLSL.
    Embedded { version: u16, is_webgl: bool },
}

impl Version {
    /// Create a new gles version
    pub const fn new_gles(version: u16) -> Self {
        Self::Embedded {
            version,
            is_webgl: false,
        }
    }

    /// Returns true if self is `Version::Embedded` (i.e. is a es version)
    pub const fn is_es(&self) -> bool {
        match *self {
            Version::Desktop(_) => false,
            Version::Embedded { .. } => true,
        }
    }

    /// Returns true if targeting WebGL
    pub const fn is_webgl(&self) -> bool {
        match *self {
            Version::Desktop(_) => false,
            Version::Embedded { is_webgl, .. } => is_webgl,
        }
    }

    /// Checks the list of currently supported versions and returns true if it contains the
    /// specified version
    ///
    /// # Notes
    /// As an invalid version number will never be added to the supported version list
    /// so this also checks for version validity
    pub fn is_supported(&self) -> bool {
        match *self {
            Version::Desktop(v) => SUPPORTED_CORE_VERSIONS.contains(&v),
            Version::Embedded { version: v, .. } => SUPPORTED_ES_VERSIONS.contains(&v),
        }
    }

    pub fn supports_io_locations(&self) -> bool {
        *self >= Version::Desktop(330) || *self >= Version::new_gles(300)
    }

    /// Checks if the version supports all of the explicit layouts:
    /// - `location=` qualifiers for bindings
    /// - `binding=` qualifiers for resources
    ///
    /// Note: `location=` for vertex inputs and fragment outputs is supported
    /// unconditionally for GLES 300.
    pub fn supports_explicit_locations(&self) -> bool {
        *self >= Version::Desktop(420) || *self >= Version::new_gles(310)
    }

    pub fn supports_early_depth_test(&self) -> bool {
        *self >= Version::Desktop(130) || *self >= Version::new_gles(310)
    }

    pub fn supports_std140_layout(&self) -> bool {
        *self >= Version::Desktop(140) || *self >= Version::new_gles(300)
    }

    pub fn supports_std430_layout(&self) -> bool {
        // std430 is available from 400 via GL_ARB_shader_storage_buffer_object.
        *self >= Version::Desktop(400) || *self >= Version::new_gles(310)
    }

    pub fn supports_fma_function(&self) -> bool {
        *self >= Version::Desktop(400) || *self >= Version::new_gles(320)
    }

    pub fn supports_integer_functions(&self) -> bool {
        *self >= Version::Desktop(400) || *self >= Version::new_gles(310)
    }

    pub fn supports_frexp_function(&self) -> bool {
        *self >= Version::Desktop(400) || *self >= Version::new_gles(310)
    }

    pub fn supports_derivative_control(&self) -> bool {
        *self >= Version::Desktop(450)
    }

    // For supports_pack_unpack_4x8, supports_pack_unpack_snorm_2x16, supports_pack_unpack_unorm_2x16
    // see:
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/unpackUnorm.xhtml
    // https://registry.khronos.org/OpenGL-Refpages/es3/html/unpackUnorm.xhtml
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/packUnorm.xhtml
    // https://registry.khronos.org/OpenGL-Refpages/es3/html/packUnorm.xhtml
    pub fn supports_pack_unpack_4x8(&self) -> bool {
        *self >= Version::Desktop(400) || *self >= Version::new_gles(310)
    }
    pub fn supports_pack_unpack_snorm_2x16(&self) -> bool {
        *self >= Version::Desktop(420) || *self >= Version::new_gles(300)
    }
    pub fn supports_pack_unpack_unorm_2x16(&self) -> bool {
        *self >= Version::Desktop(400) || *self >= Version::new_gles(300)
    }

    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/unpackHalf2x16.xhtml
    // https://registry.khronos.org/OpenGL-Refpages/gl4/html/packHalf2x16.xhtml
    // https://registry.khronos.org/OpenGL-Refpages/es3/html/unpackHalf2x16.xhtml
    // https://registry.khronos.org/OpenGL-Refpages/es3/html/packHalf2x16.xhtml
    pub fn supports_pack_unpack_half_2x16(&self) -> bool {
        *self >= Version::Desktop(420) || *self >= Version::new_gles(300)
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (*self, *other) {
            (Version::Desktop(x), Version::Desktop(y)) => Some(x.cmp(&y)),
            (Version::Embedded { version: x, .. }, Version::Embedded { version: y, .. }) => {
                Some(x.cmp(&y))
            }
            _ => None,
        }
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Version::Desktop(v) => write!(f, "{v} core"),
            Version::Embedded { version: v, .. } => write!(f, "{v} es"),
        }
    }
}

/// Mapping between resources and bindings.
pub type BindingMap = alloc::collections::BTreeMap<crate::ResourceBinding, u8>;

/// Separate type from `naga::ScalarKind` so that naga can easily add impl's
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum GlslScalarKind {
    Sint,
    Uint,
    Float,
}

/// Separate type from `naga::VectorSize` so that naga can easily add impl's
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum GlslVectorSize {
    Bi = 2,
    Tri = 3,
    Quad = 4,
}
impl GlslVectorSize {
    pub fn alignment(&self) -> u32 {
        match self {
            Self::Bi => 2,
            Self::Tri | Self::Quad => 4,
        }
    }
}

/// Separate type from `naga::Scalar` so that naga can easily add impl's
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub struct GlslScalar {
    pub kind: GlslScalarKind,
    pub width: u8,
}

impl GlslScalar {
    pub const F32: Self = Self {
        kind: GlslScalarKind::Float,
        width: 4,
    };
    pub const I32: Self = Self {
        kind: GlslScalarKind::Sint,
        width: 4,
    };
    pub const U32: Self = Self {
        kind: GlslScalarKind::Uint,
        width: 4,
    };
}

/// A subset of `naga::TypeInner` so that uniforms can be analyzed without pulling in the entire naga IR.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize))]
#[cfg_attr(feature = "deserialize", derive(serde::Deserialize))]
pub enum GlslUniformType {
    Scalar(GlslScalar),
    Vector {
        size: GlslVectorSize,
        scalar: GlslScalar,
    },
    Matrix {
        columns: GlslVectorSize,
        rows: GlslVectorSize,
        scalar: GlslScalar,
    },
}
impl GlslUniformType {
    pub fn size(&self) -> u32 {
        match self {
            Self::Scalar(scalar) => scalar.width as u32,
            Self::Vector { size, scalar } => *size as u32 * scalar.width as u32,
            // matrices are treated as arrays of aligned columns
            Self::Matrix {
                columns,
                rows,
                scalar,
            } => rows.alignment() * scalar.width as u32 * *columns as u32,
        }
    }
}

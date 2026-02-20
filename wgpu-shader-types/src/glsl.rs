use core::{cmp::Ordering, fmt};

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
        *self >= Version::Desktop(430) || *self >= Version::new_gles(310)
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

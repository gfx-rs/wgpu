//! Producing the WGSL forms of types, for use in error messages.

use crate::proc::GlobalCtx;
use crate::Handle;

impl crate::proc::TypeResolution {
    pub fn to_wgsl(&self, gctx: &GlobalCtx) -> String {
        match *self {
            crate::proc::TypeResolution::Handle(handle) => handle.to_wgsl(gctx),
            crate::proc::TypeResolution::Value(ref inner) => inner.to_wgsl(gctx),
        }
    }
}

impl Handle<crate::Type> {
    /// Formats the type as it is written in wgsl.
    ///
    /// For example `vec3<f32>`.
    pub fn to_wgsl(self, gctx: &GlobalCtx) -> String {
        let ty = &gctx.types[self];
        match ty.name {
            Some(ref name) => name.clone(),
            None => ty.inner.to_wgsl(gctx),
        }
    }
}

impl crate::TypeInner {
    /// Formats the type as it is written in wgsl.
    ///
    /// For example `vec3<f32>`.
    ///
    /// Note: `TypeInner::Struct` doesn't include the name of the
    /// struct type. Therefore this method will simply return "struct"
    /// for them.
    pub fn to_wgsl(&self, gctx: &GlobalCtx) -> String {
        use crate::TypeInner as Ti;

        match *self {
            Ti::Scalar(scalar) => scalar.to_wgsl(),
            Ti::Vector { size, scalar } => {
                format!("vec{}<{}>", size as u32, scalar.to_wgsl())
            }
            Ti::Matrix {
                columns,
                rows,
                scalar,
            } => {
                format!(
                    "mat{}x{}<{}>",
                    columns as u32,
                    rows as u32,
                    scalar.to_wgsl(),
                )
            }
            Ti::Atomic(scalar) => {
                format!("atomic<{}>", scalar.to_wgsl())
            }
            Ti::Pointer { base, .. } => {
                let name = base.to_wgsl(gctx);
                format!("ptr<{name}>")
            }
            Ti::ValuePointer { scalar, .. } => {
                format!("ptr<{}>", scalar.to_wgsl())
            }
            Ti::Array { base, size, .. } => {
                let base = base.to_wgsl(gctx);
                match size {
                    crate::ArraySize::Constant(size) => format!("array<{base}, {size}>"),
                    crate::ArraySize::Dynamic => format!("array<{base}>"),
                }
            }
            Ti::Struct { .. } => {
                // TODO: Actually output the struct?
                "struct".to_string()
            }
            Ti::Image {
                dim,
                arrayed,
                class,
            } => {
                let dim_suffix = match dim {
                    crate::ImageDimension::D1 => "_1d",
                    crate::ImageDimension::D2 => "_2d",
                    crate::ImageDimension::D3 => "_3d",
                    crate::ImageDimension::Cube => "_cube",
                };
                let array_suffix = if arrayed { "_array" } else { "" };

                let class_suffix = match class {
                    crate::ImageClass::Sampled { multi: true, .. } => "_multisampled",
                    crate::ImageClass::Depth { multi: false } => "_depth",
                    crate::ImageClass::Depth { multi: true } => "_depth_multisampled",
                    crate::ImageClass::Sampled { multi: false, .. }
                    | crate::ImageClass::Storage { .. } => "",
                };

                let type_in_brackets = match class {
                    crate::ImageClass::Sampled { kind, .. } => {
                        // Note: The only valid widths are 4 bytes wide.
                        // The lexer has already verified this, so we can safely assume it here.
                        // https://gpuweb.github.io/gpuweb/wgsl/#sampled-texture-type
                        let element_type = crate::Scalar { kind, width: 4 }.to_wgsl();
                        format!("<{element_type}>")
                    }
                    crate::ImageClass::Depth { multi: _ } => String::new(),
                    crate::ImageClass::Storage { format, access } => {
                        if access.contains(crate::StorageAccess::STORE) {
                            format!("<{},write>", format.to_wgsl())
                        } else {
                            format!("<{}>", format.to_wgsl())
                        }
                    }
                };

                format!("texture{class_suffix}{dim_suffix}{array_suffix}{type_in_brackets}")
            }
            Ti::Sampler { .. } => "sampler".to_string(),
            Ti::AccelerationStructure => "acceleration_structure".to_string(),
            Ti::RayQuery => "ray_query".to_string(),
            Ti::BindingArray { base, size, .. } => {
                let member_type = &gctx.types[base];
                let base = member_type.name.as_deref().unwrap_or("unknown");
                match size {
                    crate::ArraySize::Constant(size) => format!("binding_array<{base}, {size}>"),
                    crate::ArraySize::Dynamic => format!("binding_array<{base}>"),
                }
            }
        }
    }
}

impl crate::Scalar {
    /// Format a scalar kind+width as a type is written in wgsl.
    ///
    /// Examples: `f32`, `u64`, `bool`.
    pub fn to_wgsl(self) -> String {
        let prefix = match self.kind {
            crate::ScalarKind::Sint => "i",
            crate::ScalarKind::Uint => "u",
            crate::ScalarKind::Float => "f",
            crate::ScalarKind::Bool => return "bool".to_string(),
        };
        format!("{}{}", prefix, self.width * 8)
    }
}

impl crate::StorageFormat {
    pub const fn to_wgsl(self) -> &'static str {
        use crate::StorageFormat as Sf;
        match self {
            Sf::R8Unorm => "r8unorm",
            Sf::R8Snorm => "r8snorm",
            Sf::R8Uint => "r8uint",
            Sf::R8Sint => "r8sint",
            Sf::R16Uint => "r16uint",
            Sf::R16Sint => "r16sint",
            Sf::R16Float => "r16float",
            Sf::Rg8Unorm => "rg8unorm",
            Sf::Rg8Snorm => "rg8snorm",
            Sf::Rg8Uint => "rg8uint",
            Sf::Rg8Sint => "rg8sint",
            Sf::R32Uint => "r32uint",
            Sf::R32Sint => "r32sint",
            Sf::R32Float => "r32float",
            Sf::Rg16Uint => "rg16uint",
            Sf::Rg16Sint => "rg16sint",
            Sf::Rg16Float => "rg16float",
            Sf::Rgba8Unorm => "rgba8unorm",
            Sf::Rgba8Snorm => "rgba8snorm",
            Sf::Rgba8Uint => "rgba8uint",
            Sf::Rgba8Sint => "rgba8sint",
            Sf::Bgra8Unorm => "bgra8unorm",
            Sf::Rgb10a2Uint => "rgb10a2uint",
            Sf::Rgb10a2Unorm => "rgb10a2unorm",
            Sf::Rg11b10Float => "rg11b10float",
            Sf::Rg32Uint => "rg32uint",
            Sf::Rg32Sint => "rg32sint",
            Sf::Rg32Float => "rg32float",
            Sf::Rgba16Uint => "rgba16uint",
            Sf::Rgba16Sint => "rgba16sint",
            Sf::Rgba16Float => "rgba16float",
            Sf::Rgba32Uint => "rgba32uint",
            Sf::Rgba32Sint => "rgba32sint",
            Sf::Rgba32Float => "rgba32float",
            Sf::R16Unorm => "r16unorm",
            Sf::R16Snorm => "r16snorm",
            Sf::Rg16Unorm => "rg16unorm",
            Sf::Rg16Snorm => "rg16snorm",
            Sf::Rgba16Unorm => "rgba16unorm",
            Sf::Rgba16Snorm => "rgba16snorm",
        }
    }
}

mod tests {
    #[test]
    fn to_wgsl() {
        use std::num::NonZeroU32;

        let mut types = crate::UniqueArena::new();

        let mytype1 = types.insert(
            crate::Type {
                name: Some("MyType1".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![],
                    span: 0,
                },
            },
            Default::default(),
        );
        let mytype2 = types.insert(
            crate::Type {
                name: Some("MyType2".to_string()),
                inner: crate::TypeInner::Struct {
                    members: vec![],
                    span: 0,
                },
            },
            Default::default(),
        );

        let gctx = crate::proc::GlobalCtx {
            types: &types,
            constants: &crate::Arena::new(),
            overrides: &crate::Arena::new(),
            const_expressions: &crate::Arena::new(),
        };
        let array = crate::TypeInner::Array {
            base: mytype1,
            stride: 4,
            size: crate::ArraySize::Constant(unsafe { NonZeroU32::new_unchecked(32) }),
        };
        assert_eq!(array.to_wgsl(&gctx), "array<MyType1, 32>");

        let mat = crate::TypeInner::Matrix {
            rows: crate::VectorSize::Quad,
            columns: crate::VectorSize::Bi,
            scalar: crate::Scalar::F64,
        };
        assert_eq!(mat.to_wgsl(&gctx), "mat2x4<f64>");

        let ptr = crate::TypeInner::Pointer {
            base: mytype2,
            space: crate::AddressSpace::Storage {
                access: crate::StorageAccess::default(),
            },
        };
        assert_eq!(ptr.to_wgsl(&gctx), "ptr<MyType2>");

        let img1 = crate::TypeInner::Image {
            dim: crate::ImageDimension::D2,
            arrayed: false,
            class: crate::ImageClass::Sampled {
                kind: crate::ScalarKind::Float,
                multi: true,
            },
        };
        assert_eq!(img1.to_wgsl(&gctx), "texture_multisampled_2d<f32>");

        let img2 = crate::TypeInner::Image {
            dim: crate::ImageDimension::Cube,
            arrayed: true,
            class: crate::ImageClass::Depth { multi: false },
        };
        assert_eq!(img2.to_wgsl(&gctx), "texture_depth_cube_array");

        let img3 = crate::TypeInner::Image {
            dim: crate::ImageDimension::D2,
            arrayed: false,
            class: crate::ImageClass::Depth { multi: true },
        };
        assert_eq!(img3.to_wgsl(&gctx), "texture_depth_multisampled_2d");

        let array = crate::TypeInner::BindingArray {
            base: mytype1,
            size: crate::ArraySize::Constant(unsafe { NonZeroU32::new_unchecked(32) }),
        };
        assert_eq!(array.to_wgsl(&gctx), "binding_array<MyType1, 32>");
    }
}

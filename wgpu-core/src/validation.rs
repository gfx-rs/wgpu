/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{binding_model::BindEntryMap, FastHashMap, FastHashSet};
use naga::valid::GlobalUse;
use std::{collections::hash_map::Entry, fmt};
use thiserror::Error;
use wgt::{BindGroupLayoutEntry, BindingType};

#[derive(Debug)]
enum ResourceType {
    Buffer {
        size: wgt::BufferSize,
    },
    Texture {
        dim: naga::ImageDimension,
        arrayed: bool,
        class: naga::ImageClass,
    },
    Sampler {
        comparison: bool,
    },
}

#[derive(Debug)]
struct Resource {
    name: Option<String>,
    bind: naga::ResourceBinding,
    ty: ResourceType,
    class: naga::StorageClass,
}

#[derive(Clone, Copy, Debug)]
enum NumericDimension {
    Scalar,
    Vector(naga::VectorSize),
    Matrix(naga::VectorSize, naga::VectorSize),
}

impl fmt::Display for NumericDimension {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Scalar => write!(f, ""),
            Self::Vector(size) => write!(f, "x{}", size as u8),
            Self::Matrix(columns, rows) => write!(f, "x{}{}", columns as u8, rows as u8),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NumericType {
    dim: NumericDimension,
    kind: naga::ScalarKind,
    width: naga::Bytes,
}

impl fmt::Display for NumericType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}{}{}", self.kind, self.width * 8, self.dim)
    }
}

#[derive(Clone, Debug)]
pub struct InterfaceVar {
    pub ty: NumericType,
    interpolation: Option<naga::Interpolation>,
    sampling: Option<naga::Sampling>,
}

impl InterfaceVar {
    pub fn vertex_attribute(format: wgt::VertexFormat) -> Self {
        InterfaceVar {
            ty: NumericType::from_vertex_format(format),
            interpolation: None,
            sampling: None,
        }
    }
}

impl fmt::Display for InterfaceVar {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} interpolated as {:?} with sampling {:?}",
            self.ty, self.interpolation, self.sampling
        )
    }
}

#[derive(Debug)]
enum Varying {
    Local { location: u32, iv: InterfaceVar },
    BuiltIn(naga::BuiltIn),
}

#[derive(Debug)]
struct SpecializationConstant {
    id: u32,
    ty: NumericType,
}

#[derive(Debug, Default)]
struct EntryPoint {
    inputs: Vec<Varying>,
    outputs: Vec<Varying>,
    resources: Vec<(naga::Handle<Resource>, GlobalUse)>,
    spec_constants: Vec<SpecializationConstant>,
    sampling_pairs: FastHashSet<(naga::Handle<Resource>, naga::Handle<Resource>)>,
}

#[derive(Debug)]
pub struct Interface {
    resources: naga::Arena<Resource>,
    entry_points: FastHashMap<(naga::ShaderStage, String), EntryPoint>,
}

#[derive(Clone, Debug, Error)]
#[error("buffer usage is {actual:?} which does not contain required usage {expected:?}")]
pub struct MissingBufferUsageError {
    pub(crate) actual: wgt::BufferUsage,
    pub(crate) expected: wgt::BufferUsage,
}

/// Checks that the given buffer usage contains the required buffer usage,
/// returns an error otherwise.
pub fn check_buffer_usage(
    actual: wgt::BufferUsage,
    expected: wgt::BufferUsage,
) -> Result<(), MissingBufferUsageError> {
    if !actual.contains(expected) {
        Err(MissingBufferUsageError { actual, expected })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[error("texture usage is {actual:?} which does not contain required usage {expected:?}")]
pub struct MissingTextureUsageError {
    pub(crate) actual: wgt::TextureUsage,
    pub(crate) expected: wgt::TextureUsage,
}

/// Checks that the given texture usage contains the required texture usage,
/// returns an error otherwise.
pub fn check_texture_usage(
    actual: wgt::TextureUsage,
    expected: wgt::TextureUsage,
) -> Result<(), MissingTextureUsageError> {
    if !actual.contains(expected) {
        Err(MissingTextureUsageError { actual, expected })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
pub enum BindingError {
    #[error("binding is missing from the pipeline layout")]
    Missing,
    #[error("visibility flags don't include the shader stage")]
    Invisible,
    #[error("The shader requires the load/store access flags {required:?} but only {allowed:?} is allowed")]
    WrongUsage {
        required: GlobalUse,
        allowed: GlobalUse,
    },
    #[error("type on the shader side does not match the pipeline binding")]
    WrongType,
    #[error("buffer structure size {0}, added to one element of an unbound array, if it's the last field, ended up greater than the given `min_binding_size`")]
    WrongBufferSize(wgt::BufferSize),
    #[error("view dimension {dim:?} (is array: {is_array}) doesn't match the binding {binding:?}")]
    WrongTextureViewDimension {
        dim: naga::ImageDimension,
        is_array: bool,
        binding: BindingType,
    },
    #[error("texture class {binding:?} doesn't match the shader {shader:?}")]
    WrongTextureClass {
        binding: naga::ImageClass,
        shader: naga::ImageClass,
    },
    #[error("comparison flag doesn't match the shader")]
    WrongSamplerComparison,
    #[error("derived bind group layout type is not consistent between stages")]
    InconsistentlyDerivedType,
    #[error("texture format {0:?} is not supported for storage use")]
    BadStorageFormat(wgt::TextureFormat),
}

#[derive(Clone, Debug, Error)]
pub enum FilteringError {
    #[error("integer textures can't be sampled")]
    Integer,
    #[error("non-filterable float texture")]
    NonFilterable,
}

#[derive(Clone, Debug, Error)]
pub enum InputError {
    #[error("input is not provided by the earlier stage in the pipeline")]
    Missing,
    #[error("input type is not compatible with the provided {0}")]
    WrongType(NumericType),
    #[error("input interpolation doesn't match provided {0:?}")]
    InterpolationMismatch(Option<naga::Interpolation>),
    #[error("input sampling doesn't match provided {0:?}")]
    SamplingMismatch(Option<naga::Sampling>),
}

/// Errors produced when validating a programmable stage of a pipeline.
#[derive(Clone, Debug, Error)]
pub enum StageError {
    #[error("shader module is invalid")]
    InvalidModule,
    #[error("unable to find entry point '{0:?}'")]
    MissingEntryPoint(String),
    #[error("shader global {0:?} is not available in the layout pipeline layout")]
    Binding(naga::ResourceBinding, #[source] BindingError),
    #[error("unable to filter the texture ({texture:?}) by the sampler ({sampler:?})")]
    Filtering {
        texture: naga::ResourceBinding,
        sampler: naga::ResourceBinding,
        #[source]
        error: FilteringError,
    },
    #[error("location[{location}] {var} is not provided by the previous stage outputs")]
    Input {
        location: wgt::ShaderLocation,
        var: InterfaceVar,
        #[source]
        error: InputError,
    },
}

fn map_storage_format_to_naga(format: wgt::TextureFormat) -> Option<naga::StorageFormat> {
    use naga::StorageFormat as Sf;
    use wgt::TextureFormat as Tf;
    // Using the table in https://gpuweb.github.io/gpuweb/#plain-color-formats
    Some(match format {
        Tf::R32Uint => Sf::R32Uint,
        Tf::R32Sint => Sf::R32Sint,
        Tf::R32Float => Sf::R32Float,
        Tf::Rgba8Unorm => Sf::Rgba8Unorm,
        Tf::Rgba8Snorm => Sf::Rgba8Snorm,
        Tf::Rgba8Uint => Sf::Rgba8Uint,
        Tf::Rgba8Sint => Sf::Rgba8Sint,
        Tf::Rg32Uint => Sf::Rg32Uint,
        Tf::Rg32Sint => Sf::Rg32Sint,
        Tf::Rg32Float => Sf::Rg32Float,
        Tf::Rgba16Uint => Sf::Rgba16Uint,
        Tf::Rgba16Sint => Sf::Rgba16Sint,
        Tf::Rgba16Float => Sf::Rgba16Float,
        Tf::Rgba32Uint => Sf::Rgba32Uint,
        Tf::Rgba32Sint => Sf::Rgba32Sint,
        Tf::Rgba32Float => Sf::Rgba32Float,
        _ => return None,
    })
}

fn map_storage_format_from_naga(format: naga::StorageFormat) -> wgt::TextureFormat {
    use naga::StorageFormat as Sf;
    use wgt::TextureFormat as Tf;
    match format {
        Sf::R8Unorm => Tf::R8Unorm,
        Sf::R8Snorm => Tf::R8Snorm,
        Sf::R8Uint => Tf::R8Uint,
        Sf::R8Sint => Tf::R8Sint,
        Sf::R16Uint => Tf::R16Uint,
        Sf::R16Sint => Tf::R16Sint,
        Sf::R16Float => Tf::R16Float,
        Sf::Rg8Unorm => Tf::Rg8Unorm,
        Sf::Rg8Snorm => Tf::Rg8Snorm,
        Sf::Rg8Uint => Tf::Rg8Uint,
        Sf::Rg8Sint => Tf::Rg8Sint,
        Sf::R32Uint => Tf::R32Uint,
        Sf::R32Sint => Tf::R32Sint,
        Sf::R32Float => Tf::R32Float,
        Sf::Rg16Uint => Tf::Rg16Uint,
        Sf::Rg16Sint => Tf::Rg16Sint,
        Sf::Rg16Float => Tf::Rg16Float,
        Sf::Rgba8Unorm => Tf::Rgba8Unorm,
        Sf::Rgba8Snorm => Tf::Rgba8Snorm,
        Sf::Rgba8Uint => Tf::Rgba8Uint,
        Sf::Rgba8Sint => Tf::Rgba8Sint,
        Sf::Rgb10a2Unorm => Tf::Rgb10a2Unorm,
        Sf::Rg11b10Float => Tf::Rg11b10Float,
        Sf::Rg32Uint => Tf::Rg32Uint,
        Sf::Rg32Sint => Tf::Rg32Sint,
        Sf::Rg32Float => Tf::Rg32Float,
        Sf::Rgba16Uint => Tf::Rgba16Uint,
        Sf::Rgba16Sint => Tf::Rgba16Sint,
        Sf::Rgba16Float => Tf::Rgba16Float,
        Sf::Rgba32Uint => Tf::Rgba32Uint,
        Sf::Rgba32Sint => Tf::Rgba32Sint,
        Sf::Rgba32Float => Tf::Rgba32Float,
    }
}

impl Resource {
    fn check_binding_use(
        &self,
        entry: &BindGroupLayoutEntry,
        shader_usage: GlobalUse,
    ) -> Result<(), BindingError> {
        let allowed_usage = match self.ty {
            ResourceType::Buffer { size } => {
                let (allowed_usage, min_size) = match entry.ty {
                    BindingType::Buffer {
                        ty,
                        has_dynamic_offset: _,
                        min_binding_size,
                    } => {
                        let global_use = match ty {
                            wgt::BufferBindingType::Uniform
                            | wgt::BufferBindingType::Storage { read_only: true } => {
                                GlobalUse::READ | GlobalUse::QUERY
                            }
                            wgt::BufferBindingType::Storage { read_only: _ } => GlobalUse::all(),
                        };
                        (global_use, min_binding_size)
                    }
                    _ => return Err(BindingError::WrongType),
                };
                match min_size {
                    Some(non_zero) if non_zero < size => {
                        return Err(BindingError::WrongBufferSize(size))
                    }
                    _ => (),
                }
                allowed_usage
            }
            ResourceType::Sampler { comparison } => match entry.ty {
                BindingType::Sampler {
                    filtering: _,
                    comparison: cmp,
                } => {
                    if cmp == comparison {
                        GlobalUse::READ
                    } else {
                        return Err(BindingError::WrongSamplerComparison);
                    }
                }
                _ => return Err(BindingError::WrongType),
            },
            ResourceType::Texture {
                dim,
                arrayed,
                class,
            } => {
                let view_dimension = match entry.ty {
                    BindingType::Texture { view_dimension, .. }
                    | BindingType::StorageTexture { view_dimension, .. } => view_dimension,
                    _ => {
                        return Err(BindingError::WrongTextureViewDimension {
                            dim,
                            is_array: false,
                            binding: entry.ty,
                        })
                    }
                };
                if arrayed {
                    match (dim, view_dimension) {
                        (naga::ImageDimension::D2, wgt::TextureViewDimension::D2Array) => (),
                        (naga::ImageDimension::Cube, wgt::TextureViewDimension::CubeArray) => (),
                        _ => {
                            return Err(BindingError::WrongTextureViewDimension {
                                dim,
                                is_array: true,
                                binding: entry.ty,
                            })
                        }
                    }
                } else {
                    match (dim, view_dimension) {
                        (naga::ImageDimension::D1, wgt::TextureViewDimension::D1) => (),
                        (naga::ImageDimension::D2, wgt::TextureViewDimension::D2) => (),
                        (naga::ImageDimension::D3, wgt::TextureViewDimension::D3) => (),
                        (naga::ImageDimension::Cube, wgt::TextureViewDimension::Cube) => (),
                        _ => {
                            return Err(BindingError::WrongTextureViewDimension {
                                dim,
                                is_array: false,
                                binding: entry.ty,
                            })
                        }
                    }
                }
                let (expected_class, usage) = match entry.ty {
                    BindingType::Texture {
                        sample_type,
                        view_dimension: _,
                        multisampled: multi,
                    } => {
                        let class = match sample_type {
                            wgt::TextureSampleType::Float { .. } => naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Float,
                                multi,
                            },
                            wgt::TextureSampleType::Sint => naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Sint,
                                multi,
                            },
                            wgt::TextureSampleType::Uint => naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Uint,
                                multi,
                            },
                            wgt::TextureSampleType::Depth => naga::ImageClass::Depth,
                        };
                        (class, GlobalUse::READ | GlobalUse::QUERY)
                    }
                    BindingType::StorageTexture {
                        access,
                        format,
                        view_dimension: _,
                    } => {
                        let naga_format = map_storage_format_to_naga(format)
                            .ok_or(BindingError::BadStorageFormat(format))?;
                        let usage = match access {
                            wgt::StorageTextureAccess::ReadOnly => {
                                GlobalUse::READ | GlobalUse::QUERY
                            }
                            wgt::StorageTextureAccess::WriteOnly => {
                                GlobalUse::WRITE | GlobalUse::QUERY
                            }
                            wgt::StorageTextureAccess::ReadWrite => GlobalUse::all(),
                        };
                        (naga::ImageClass::Storage(naga_format), usage)
                    }
                    _ => return Err(BindingError::WrongType),
                };
                if class != expected_class {
                    return Err(BindingError::WrongTextureClass {
                        binding: expected_class,
                        shader: class,
                    });
                }
                usage
            }
        };

        if allowed_usage.contains(shader_usage) {
            Ok(())
        } else {
            Err(BindingError::WrongUsage {
                required: shader_usage,
                allowed: allowed_usage,
            })
        }
    }

    fn derive_binding_type(&self, shader_usage: GlobalUse) -> Result<BindingType, BindingError> {
        Ok(match self.ty {
            ResourceType::Buffer { size } => BindingType::Buffer {
                ty: match self.class {
                    naga::StorageClass::Uniform => wgt::BufferBindingType::Uniform,
                    naga::StorageClass::Storage => wgt::BufferBindingType::Storage {
                        read_only: !shader_usage.contains(GlobalUse::WRITE),
                    },
                    _ => return Err(BindingError::WrongType),
                },
                has_dynamic_offset: false,
                min_binding_size: Some(size),
            },
            ResourceType::Sampler { comparison } => BindingType::Sampler {
                filtering: true,
                comparison,
            },
            ResourceType::Texture {
                dim,
                arrayed,
                class,
            } => {
                let view_dimension = match dim {
                    naga::ImageDimension::D1 => wgt::TextureViewDimension::D1,
                    naga::ImageDimension::D2 if arrayed => wgt::TextureViewDimension::D2Array,
                    naga::ImageDimension::D2 => wgt::TextureViewDimension::D2,
                    naga::ImageDimension::D3 => wgt::TextureViewDimension::D3,
                    naga::ImageDimension::Cube if arrayed => wgt::TextureViewDimension::CubeArray,
                    naga::ImageDimension::Cube => wgt::TextureViewDimension::Cube,
                };
                match class {
                    naga::ImageClass::Sampled { multi, kind } => BindingType::Texture {
                        sample_type: match kind {
                            naga::ScalarKind::Float => {
                                wgt::TextureSampleType::Float { filterable: true }
                            }
                            naga::ScalarKind::Sint => wgt::TextureSampleType::Sint,
                            naga::ScalarKind::Uint => wgt::TextureSampleType::Uint,
                            naga::ScalarKind::Bool => unreachable!(),
                        },
                        view_dimension,
                        multisampled: multi,
                    },
                    naga::ImageClass::Depth => BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Depth,
                        view_dimension,
                        multisampled: false,
                    },
                    naga::ImageClass::Storage(format) => BindingType::StorageTexture {
                        access: if shader_usage.contains(GlobalUse::WRITE) {
                            wgt::StorageTextureAccess::WriteOnly
                        } else {
                            wgt::StorageTextureAccess::ReadOnly
                        },
                        view_dimension,
                        format: {
                            let f = map_storage_format_from_naga(format);
                            let original = map_storage_format_to_naga(f)
                                .ok_or(BindingError::BadStorageFormat(f))?;
                            debug_assert_eq!(format, original);
                            f
                        },
                    },
                }
            }
        })
    }
}

impl NumericType {
    fn from_vertex_format(format: wgt::VertexFormat) -> Self {
        use naga::{ScalarKind as Sk, VectorSize as Vs};
        use wgt::VertexFormat as Vf;

        let (dim, kind, width) = match format {
            Vf::Uint32 => (NumericDimension::Scalar, Sk::Uint, 4),
            Vf::Uint8x2 | Vf::Uint16x2 | Vf::Uint32x2 => {
                (NumericDimension::Vector(Vs::Bi), Sk::Uint, 4)
            }
            Vf::Uint32x3 => (NumericDimension::Vector(Vs::Tri), Sk::Uint, 4),
            Vf::Uint8x4 | Vf::Uint16x4 | Vf::Uint32x4 => {
                (NumericDimension::Vector(Vs::Quad), Sk::Uint, 4)
            }
            Vf::Sint32 => (NumericDimension::Scalar, Sk::Sint, 4),
            Vf::Sint8x2 | Vf::Sint16x2 | Vf::Sint32x2 => {
                (NumericDimension::Vector(Vs::Bi), Sk::Sint, 4)
            }
            Vf::Sint32x3 => (NumericDimension::Vector(Vs::Tri), Sk::Sint, 4),
            Vf::Sint8x4 | Vf::Sint16x4 | Vf::Sint32x4 => {
                (NumericDimension::Vector(Vs::Quad), Sk::Sint, 4)
            }
            Vf::Float32 => (NumericDimension::Scalar, Sk::Float, 4),
            Vf::Unorm8x2
            | Vf::Snorm8x2
            | Vf::Unorm16x2
            | Vf::Snorm16x2
            | Vf::Float16x2
            | Vf::Float32x2 => (NumericDimension::Vector(Vs::Bi), Sk::Float, 4),
            Vf::Float32x3 => (NumericDimension::Vector(Vs::Tri), Sk::Float, 4),
            Vf::Unorm8x4
            | Vf::Snorm8x4
            | Vf::Unorm16x4
            | Vf::Snorm16x4
            | Vf::Float16x4
            | Vf::Float32x4 => (NumericDimension::Vector(Vs::Quad), Sk::Float, 4),
            Vf::Float64 => (NumericDimension::Scalar, Sk::Float, 8),
            Vf::Float64x2 => (NumericDimension::Vector(Vs::Bi), Sk::Float, 8),
            Vf::Float64x3 => (NumericDimension::Vector(Vs::Tri), Sk::Float, 8),
            Vf::Float64x4 => (NumericDimension::Vector(Vs::Quad), Sk::Float, 8),
        };

        NumericType {
            dim,
            kind,
            //Note: Shader always sees data as int, uint, or float.
            // It doesn't know if the original is normalized in a tighter form.
            width,
        }
    }

    fn from_texture_format(format: wgt::TextureFormat) -> Self {
        use naga::{ScalarKind as Sk, VectorSize as Vs};
        use wgt::TextureFormat as Tf;

        let (dim, kind) = match format {
            Tf::R8Unorm | Tf::R8Snorm | Tf::R16Float | Tf::R32Float => {
                (NumericDimension::Scalar, Sk::Float)
            }
            Tf::R8Uint | Tf::R16Uint | Tf::R32Uint => (NumericDimension::Scalar, Sk::Uint),
            Tf::R8Sint | Tf::R16Sint | Tf::R32Sint => (NumericDimension::Scalar, Sk::Sint),
            Tf::Rg8Unorm | Tf::Rg8Snorm | Tf::Rg16Float | Tf::Rg32Float => {
                (NumericDimension::Vector(Vs::Bi), Sk::Float)
            }
            Tf::Rg8Uint | Tf::Rg16Uint | Tf::Rg32Uint => {
                (NumericDimension::Vector(Vs::Bi), Sk::Uint)
            }
            Tf::Rg8Sint | Tf::Rg16Sint | Tf::Rg32Sint => {
                (NumericDimension::Vector(Vs::Bi), Sk::Sint)
            }
            Tf::Rgba8Unorm
            | Tf::Rgba8UnormSrgb
            | Tf::Rgba8Snorm
            | Tf::Bgra8Unorm
            | Tf::Bgra8UnormSrgb
            | Tf::Rgb10a2Unorm
            | Tf::Rgba16Float
            | Tf::Rgba32Float => (NumericDimension::Vector(Vs::Quad), Sk::Float),
            Tf::Rgba8Uint | Tf::Rgba16Uint | Tf::Rgba32Uint => {
                (NumericDimension::Vector(Vs::Quad), Sk::Uint)
            }
            Tf::Rgba8Sint | Tf::Rgba16Sint | Tf::Rgba32Sint => {
                (NumericDimension::Vector(Vs::Quad), Sk::Sint)
            }
            Tf::Rg11b10Float => (NumericDimension::Vector(Vs::Tri), Sk::Float),
            Tf::Depth32Float | Tf::Depth24Plus | Tf::Depth24PlusStencil8 => {
                panic!("Unexpected depth format")
            }
            Tf::Bc1RgbaUnorm
            | Tf::Bc1RgbaUnormSrgb
            | Tf::Bc2RgbaUnorm
            | Tf::Bc2RgbaUnormSrgb
            | Tf::Bc3RgbaUnorm
            | Tf::Bc3RgbaUnormSrgb
            | Tf::Bc7RgbaUnorm
            | Tf::Bc7RgbaUnormSrgb
            | Tf::Etc2RgbA1Unorm
            | Tf::Etc2RgbA1UnormSrgb
            | Tf::Etc2RgbA8Unorm
            | Tf::Etc2RgbA8UnormSrgb
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
            | Tf::Astc12x12RgbaUnormSrgb => (NumericDimension::Vector(Vs::Quad), Sk::Float),
            Tf::Bc4RUnorm | Tf::Bc4RSnorm | Tf::EacRUnorm | Tf::EacRSnorm => {
                (NumericDimension::Scalar, Sk::Float)
            }
            Tf::Bc5RgUnorm | Tf::Bc5RgSnorm | Tf::EtcRgUnorm | Tf::EtcRgSnorm => {
                (NumericDimension::Vector(Vs::Bi), Sk::Float)
            }
            Tf::Bc6hRgbUfloat | Tf::Bc6hRgbSfloat | Tf::Etc2RgbUnorm | Tf::Etc2RgbUnormSrgb => {
                (NumericDimension::Vector(Vs::Tri), Sk::Float)
            }
        };

        NumericType {
            dim,
            kind,
            //Note: Shader always sees data as int, uint, or float.
            // It doesn't know if the original is normalized in a tighter form.
            width: 4,
        }
    }

    fn is_subtype_of(&self, other: &NumericType) -> bool {
        if self.width > other.width {
            return false;
        }
        if self.kind != other.kind {
            return false;
        }
        match (self.dim, other.dim) {
            (NumericDimension::Scalar, NumericDimension::Scalar) => true,
            (NumericDimension::Scalar, NumericDimension::Vector(_)) => true,
            (NumericDimension::Vector(s0), NumericDimension::Vector(s1)) => s0 <= s1,
            (NumericDimension::Matrix(c0, r0), NumericDimension::Matrix(c1, r1)) => {
                c0 == c1 && r0 == r1
            }
            _ => false,
        }
    }

    fn is_compatible_with(&self, other: &NumericType) -> bool {
        if self.kind != other.kind {
            return false;
        }
        match (self.dim, other.dim) {
            (NumericDimension::Scalar, NumericDimension::Scalar) => true,
            (NumericDimension::Scalar, NumericDimension::Vector(_)) => true,
            (NumericDimension::Vector(_), NumericDimension::Vector(_)) => true,
            (NumericDimension::Matrix(..), NumericDimension::Matrix(..)) => true,
            _ => false,
        }
    }
}

/// Return true if the fragment `format` is covered by the provided `output`.
pub fn check_texture_format(
    format: wgt::TextureFormat,
    output: &NumericType,
) -> Result<(), NumericType> {
    let nt = NumericType::from_texture_format(format);
    if nt.is_subtype_of(output) {
        Ok(())
    } else {
        Err(nt)
    }
}

pub type StageIo = FastHashMap<wgt::ShaderLocation, InterfaceVar>;

impl Interface {
    fn populate(
        list: &mut Vec<Varying>,
        binding: Option<&naga::Binding>,
        ty: naga::Handle<naga::Type>,
        arena: &naga::Arena<naga::Type>,
    ) {
        let numeric_ty = match arena[ty].inner {
            naga::TypeInner::Scalar { kind, width } => NumericType {
                dim: NumericDimension::Scalar,
                kind,
                width,
            },
            naga::TypeInner::Vector { size, kind, width } => NumericType {
                dim: NumericDimension::Vector(size),
                kind,
                width,
            },
            naga::TypeInner::Matrix {
                columns,
                rows,
                width,
            } => NumericType {
                dim: NumericDimension::Matrix(columns, rows),
                kind: naga::ScalarKind::Float,
                width,
            },
            naga::TypeInner::Struct { ref members, .. } => {
                for member in members {
                    Self::populate(list, member.binding.as_ref(), member.ty, arena);
                }
                return;
            }
            ref other => {
                log::error!("Unexpected varying type: {:?}", other);
                return;
            }
        };

        let varying = match binding {
            Some(&naga::Binding::Location {
                location,
                interpolation,
                sampling,
            }) => Varying::Local {
                location,
                iv: InterfaceVar {
                    ty: numeric_ty,
                    interpolation,
                    sampling,
                },
            },
            Some(&naga::Binding::BuiltIn(built_in)) => Varying::BuiltIn(built_in),
            None => {
                log::error!("Missing binding for a varying");
                return;
            }
        };
        list.push(varying);
    }

    pub fn new(module: &naga::Module, info: &naga::valid::ModuleInfo) -> Self {
        let mut resources = naga::Arena::new();
        let mut resource_mapping = FastHashMap::default();
        for (var_handle, var) in module.global_variables.iter() {
            let bind = match var.binding {
                Some(ref br) => br.clone(),
                _ => continue,
            };
            let ty = match module.types[var.ty].inner {
                naga::TypeInner::Struct {
                    level: naga::StructLevel::Root,
                    members: _,
                    span,
                } => ResourceType::Buffer {
                    size: wgt::BufferSize::new(span as u64).unwrap(),
                },
                naga::TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => ResourceType::Texture {
                    dim,
                    arrayed,
                    class,
                },
                naga::TypeInner::Sampler { comparison } => ResourceType::Sampler { comparison },
                ref other => {
                    log::error!("Unexpected resource type: {:?}", other);
                    continue;
                }
            };
            let handle = resources.append(Resource {
                name: var.name.clone(),
                bind,
                ty,
                class: var.class,
            });
            resource_mapping.insert(var_handle, handle);
        }

        let mut entry_points = FastHashMap::default();
        entry_points.reserve(module.entry_points.len());
        for (index, entry_point) in (&module.entry_points).iter().enumerate() {
            let info = info.get_entry_point(index);
            let mut ep = EntryPoint::default();
            for arg in entry_point.function.arguments.iter() {
                Self::populate(&mut ep.inputs, arg.binding.as_ref(), arg.ty, &module.types);
            }
            if let Some(ref result) = entry_point.function.result {
                Self::populate(
                    &mut ep.outputs,
                    result.binding.as_ref(),
                    result.ty,
                    &module.types,
                );
            }

            for (var_handle, var) in module.global_variables.iter() {
                let usage = info[var_handle];
                if usage.is_empty() {
                    continue;
                }
                if var.binding.is_some() {
                    ep.resources.push((resource_mapping[&var_handle], usage));
                }
            }

            entry_points.insert((entry_point.stage, entry_point.name.clone()), ep);
        }

        Interface {
            resources,
            entry_points,
        }
    }

    pub fn check_stage(
        &self,
        given_layouts: Option<&[&BindEntryMap]>,
        derived_layouts: &mut [BindEntryMap],
        entry_point_name: &str,
        stage_bit: wgt::ShaderStage,
        inputs: StageIo,
    ) -> Result<StageIo, StageError> {
        // Since a shader module can have multiple entry points with the same name,
        // we need to look for one with the right execution model.
        let shader_stage = match stage_bit {
            wgt::ShaderStage::VERTEX => naga::ShaderStage::Vertex,
            wgt::ShaderStage::FRAGMENT => naga::ShaderStage::Fragment,
            wgt::ShaderStage::COMPUTE => naga::ShaderStage::Compute,
            _ => unreachable!(),
        };
        let pair = (shader_stage, entry_point_name.to_string());
        let entry_point = self
            .entry_points
            .get(&pair)
            .ok_or(StageError::MissingEntryPoint(pair.1))?;

        // check resources visibility
        for &(handle, usage) in entry_point.resources.iter() {
            let res = &self.resources[handle];
            let result = match given_layouts {
                Some(layouts) => layouts
                    .get(res.bind.group as usize)
                    .and_then(|map| map.get(&res.bind.binding))
                    .ok_or(BindingError::Missing)
                    .and_then(|entry| {
                        if entry.visibility.contains(stage_bit) {
                            Ok(entry)
                        } else {
                            Err(BindingError::Invisible)
                        }
                    })
                    .and_then(|entry| res.check_binding_use(entry, usage)),
                None => derived_layouts
                    .get_mut(res.bind.group as usize)
                    .ok_or(BindingError::Missing)
                    .and_then(|set| {
                        let ty = res.derive_binding_type(usage)?;
                        match set.entry(res.bind.binding) {
                            Entry::Occupied(e) if e.get().ty != ty => {
                                return Err(BindingError::InconsistentlyDerivedType)
                            }
                            Entry::Occupied(e) => {
                                e.into_mut().visibility |= stage_bit;
                            }
                            Entry::Vacant(e) => {
                                e.insert(BindGroupLayoutEntry {
                                    binding: res.bind.binding,
                                    ty,
                                    visibility: stage_bit,
                                    count: None,
                                });
                            }
                        }
                        Ok(())
                    }),
            };
            if let Err(error) = result {
                return Err(StageError::Binding(res.bind.clone(), error));
            }
        }

        // check the compatibility between textures and samplers
        if let Some(layouts) = given_layouts {
            for &(texture_handle, sampler_handle) in entry_point.sampling_pairs.iter() {
                let texture_bind = &self.resources[texture_handle].bind;
                let sampler_bind = &self.resources[sampler_handle].bind;
                let texture_layout = &layouts[texture_bind.group as usize][&texture_bind.binding];
                let sampler_layout = &layouts[sampler_bind.group as usize][&sampler_bind.binding];
                assert!(texture_layout.visibility.contains(stage_bit));
                assert!(sampler_layout.visibility.contains(stage_bit));

                let error = match texture_layout.ty {
                    wgt::BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Float { filterable },
                        ..
                    } => match sampler_layout.ty {
                        wgt::BindingType::Sampler {
                            filtering: true, ..
                        } if !filterable => Some(FilteringError::NonFilterable),
                        _ => None,
                    },
                    wgt::BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Sint,
                        ..
                    }
                    | wgt::BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Uint,
                        ..
                    } => Some(FilteringError::Integer),
                    _ => None, // unreachable, really
                };

                if let Some(error) = error {
                    return Err(StageError::Filtering {
                        texture: texture_bind.clone(),
                        sampler: sampler_bind.clone(),
                        error,
                    });
                }
            }
        }

        // check inputs compatibility
        for input in entry_point.inputs.iter() {
            match *input {
                Varying::Local { location, ref iv } => {
                    let result =
                        inputs
                            .get(&location)
                            .ok_or(InputError::Missing)
                            .and_then(|provided| {
                                let compatible = match shader_stage {
                                    // For vertex attributes, there are defaults filled out
                                    // by the driver if data is not provided.
                                    naga::ShaderStage::Vertex => {
                                        iv.ty.is_compatible_with(&provided.ty)
                                    }
                                    naga::ShaderStage::Fragment => {
                                        if iv.interpolation != provided.interpolation {
                                            return Err(InputError::InterpolationMismatch(
                                                provided.interpolation,
                                            ));
                                        }
                                        if iv.sampling != provided.sampling {
                                            return Err(InputError::SamplingMismatch(
                                                provided.sampling,
                                            ));
                                        }
                                        iv.ty.is_subtype_of(&provided.ty)
                                    }
                                    naga::ShaderStage::Compute => false,
                                };
                                if compatible {
                                    Ok(())
                                } else {
                                    Err(InputError::WrongType(provided.ty))
                                }
                            });
                    if let Err(error) = result {
                        return Err(StageError::Input {
                            location,
                            var: iv.clone(),
                            error,
                        });
                    }
                }
                Varying::BuiltIn(_) => {}
            }
        }

        let outputs = entry_point
            .outputs
            .iter()
            .filter_map(|output| match *output {
                Varying::Local { location, ref iv } => Some((location, iv.clone())),
                Varying::BuiltIn(_) => None,
            })
            .collect();
        Ok(outputs)
    }
}

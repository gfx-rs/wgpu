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
    #[allow(unused)]
    name: Option<String>,
    bind: naga::ResourceBinding,
    ty: ResourceType,
    class: naga::AddressSpace,
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

impl NumericDimension {
    fn num_components(&self) -> u32 {
        match *self {
            Self::Scalar => 1,
            Self::Vector(size) => size as u32,
            Self::Matrix(w, h) => w as u32 * h as u32,
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

#[allow(unused)]
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
    #[allow(unused)]
    spec_constants: Vec<SpecializationConstant>,
    sampling_pairs: FastHashSet<(naga::Handle<Resource>, naga::Handle<Resource>)>,
    workgroup_size: [u32; 3],
}

#[derive(Debug)]
pub struct Interface {
    features: wgt::Features,
    limits: wgt::Limits,
    resources: naga::Arena<Resource>,
    entry_points: FastHashMap<(naga::ShaderStage, String), EntryPoint>,
}

#[derive(Clone, Debug, Error)]
#[error("Buffer usage is {actual:?} which does not contain required usage {expected:?}")]
pub struct MissingBufferUsageError {
    pub(crate) actual: wgt::BufferUsages,
    pub(crate) expected: wgt::BufferUsages,
}

/// Checks that the given buffer usage contains the required buffer usage,
/// returns an error otherwise.
pub fn check_buffer_usage(
    actual: wgt::BufferUsages,
    expected: wgt::BufferUsages,
) -> Result<(), MissingBufferUsageError> {
    if !actual.contains(expected) {
        Err(MissingBufferUsageError { actual, expected })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[error("Texture usage is {actual:?} which does not contain required usage {expected:?}")]
pub struct MissingTextureUsageError {
    pub(crate) actual: wgt::TextureUsages,
    pub(crate) expected: wgt::TextureUsages,
}

/// Checks that the given texture usage contains the required texture usage,
/// returns an error otherwise.
pub fn check_texture_usage(
    actual: wgt::TextureUsages,
    expected: wgt::TextureUsages,
) -> Result<(), MissingTextureUsageError> {
    if !actual.contains(expected) {
        Err(MissingTextureUsageError { actual, expected })
    } else {
        Ok(())
    }
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum BindingError {
    #[error("Binding is missing from the pipeline layout")]
    Missing,
    #[error("Visibility flags don't include the shader stage")]
    Invisible,
    #[error("The shader requires the load/store access flags {required:?} but only {allowed:?} is allowed")]
    WrongUsage {
        required: GlobalUse,
        allowed: GlobalUse,
    },
    #[error("Type on the shader side does not match the pipeline binding")]
    WrongType,
    #[error("Storage class {binding:?} doesn't match the shader {shader:?}")]
    WrongAddressSpace {
        binding: naga::AddressSpace,
        shader: naga::AddressSpace,
    },
    #[error("Buffer structure size {0}, added to one element of an unbound array, if it's the last field, ended up greater than the given `min_binding_size`")]
    WrongBufferSize(wgt::BufferSize),
    #[error("View dimension {dim:?} (is array: {is_array}) doesn't match the binding {binding:?}")]
    WrongTextureViewDimension {
        dim: naga::ImageDimension,
        is_array: bool,
        binding: BindingType,
    },
    #[error("Texture class {binding:?} doesn't match the shader {shader:?}")]
    WrongTextureClass {
        binding: naga::ImageClass,
        shader: naga::ImageClass,
    },
    #[error("Comparison flag doesn't match the shader")]
    WrongSamplerComparison,
    #[error("Derived bind group layout type is not consistent between stages")]
    InconsistentlyDerivedType,
    #[error("Texture format {0:?} is not supported for storage use")]
    BadStorageFormat(wgt::TextureFormat),
    #[error(
        "Storage texture usage {0:?} doesn't have a matching supported `StorageTextureAccess`"
    )]
    UnsupportedTextureStorageAccess(GlobalUse),
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum FilteringError {
    #[error("Integer textures can't be sampled with a filtering sampler")]
    Integer,
    #[error("Non-filterable float textures can't be sampled with a filtering sampler")]
    Float,
}

#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum InputError {
    #[error("Input is not provided by the earlier stage in the pipeline")]
    Missing,
    #[error("Input type is not compatible with the provided {0}")]
    WrongType(NumericType),
    #[error("Input interpolation doesn't match provided {0:?}")]
    InterpolationMismatch(Option<naga::Interpolation>),
    #[error("Input sampling doesn't match provided {0:?}")]
    SamplingMismatch(Option<naga::Sampling>),
}

/// Errors produced when validating a programmable stage of a pipeline.
#[derive(Clone, Debug, Error)]
#[non_exhaustive]
pub enum StageError {
    #[error("Shader module is invalid")]
    InvalidModule,
    #[error(
        "Shader entry point's workgroup size {current:?} ({current_total} total invocations) must be less or equal to the per-dimension limit {limit:?} and the total invocation limit {total}"
    )]
    InvalidWorkgroupSize {
        current: [u32; 3],
        current_total: u32,
        limit: [u32; 3],
        total: u32,
    },
    #[error("Shader uses {used} inter-stage components above the limit of {limit}")]
    TooManyVaryings { used: u32, limit: u32 },
    #[error("Unable to find entry point '{0}'")]
    MissingEntryPoint(String),
    #[error("Shader global {0:?} is not available in the layout pipeline layout")]
    Binding(naga::ResourceBinding, #[source] BindingError),
    #[error("Unable to filter the texture ({texture:?}) by the sampler ({sampler:?})")]
    Filtering {
        texture: naga::ResourceBinding,
        sampler: naga::ResourceBinding,
        #[source]
        error: FilteringError,
    },
    #[error("Location[{location}] {var} is not provided by the previous stage outputs")]
    Input {
        location: wgt::ShaderLocation,
        var: InterfaceVar,
        #[source]
        error: InputError,
    },
    #[error("Location[{location}] is provided by the previous stage output but is not consumed as input by this stage.")]
    InputNotConsumed { location: wgt::ShaderLocation },
}

fn map_storage_format_to_naga(format: wgt::TextureFormat) -> Option<naga::StorageFormat> {
    use naga::StorageFormat as Sf;
    use wgt::TextureFormat as Tf;

    Some(match format {
        Tf::R8Unorm => Sf::R8Unorm,
        Tf::R8Snorm => Sf::R8Snorm,
        Tf::R8Uint => Sf::R8Uint,
        Tf::R8Sint => Sf::R8Sint,

        Tf::R16Uint => Sf::R16Uint,
        Tf::R16Sint => Sf::R16Sint,
        Tf::R16Float => Sf::R16Float,
        Tf::Rg8Unorm => Sf::Rg8Unorm,
        Tf::Rg8Snorm => Sf::Rg8Snorm,
        Tf::Rg8Uint => Sf::Rg8Uint,
        Tf::Rg8Sint => Sf::Rg8Sint,

        Tf::R32Uint => Sf::R32Uint,
        Tf::R32Sint => Sf::R32Sint,
        Tf::R32Float => Sf::R32Float,
        Tf::Rg16Uint => Sf::Rg16Uint,
        Tf::Rg16Sint => Sf::Rg16Sint,
        Tf::Rg16Float => Sf::Rg16Float,
        Tf::Rgba8Unorm => Sf::Rgba8Unorm,
        Tf::Rgba8Snorm => Sf::Rgba8Snorm,
        Tf::Rgba8Uint => Sf::Rgba8Uint,
        Tf::Rgba8Sint => Sf::Rgba8Sint,

        Tf::Rgb10a2Unorm => Sf::Rgb10a2Unorm,
        Tf::Rg11b10Float => Sf::Rg11b10Float,

        Tf::Rg32Uint => Sf::Rg32Uint,
        Tf::Rg32Sint => Sf::Rg32Sint,
        Tf::Rg32Float => Sf::Rg32Float,
        Tf::Rgba16Uint => Sf::Rgba16Uint,
        Tf::Rgba16Sint => Sf::Rgba16Sint,
        Tf::Rgba16Float => Sf::Rgba16Float,

        Tf::Rgba32Uint => Sf::Rgba32Uint,
        Tf::Rgba32Sint => Sf::Rgba32Sint,
        Tf::Rgba32Float => Sf::Rgba32Float,

        Tf::R16Unorm => Sf::R16Unorm,
        Tf::R16Snorm => Sf::R16Snorm,
        Tf::Rg16Unorm => Sf::Rg16Unorm,
        Tf::Rg16Snorm => Sf::Rg16Snorm,
        Tf::Rgba16Unorm => Sf::Rgba16Unorm,
        Tf::Rgba16Snorm => Sf::Rgba16Snorm,

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

        Sf::R16Unorm => Tf::R16Unorm,
        Sf::R16Snorm => Tf::R16Snorm,
        Sf::Rg16Unorm => Tf::Rg16Unorm,
        Sf::Rg16Snorm => Tf::Rg16Snorm,
        Sf::Rgba16Unorm => Tf::Rgba16Unorm,
        Sf::Rgba16Snorm => Tf::Rgba16Snorm,
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
                        let (class, global_use) = match ty {
                            wgt::BufferBindingType::Uniform => {
                                (naga::AddressSpace::Uniform, GlobalUse::READ)
                            }
                            wgt::BufferBindingType::Storage { read_only } => {
                                let mut global_use = GlobalUse::READ | GlobalUse::QUERY;
                                global_use.set(GlobalUse::WRITE, !read_only);
                                let mut naga_access = naga::StorageAccess::LOAD;
                                naga_access.set(naga::StorageAccess::STORE, !read_only);
                                (
                                    naga::AddressSpace::Storage {
                                        access: naga_access,
                                    },
                                    global_use,
                                )
                            }
                        };
                        if self.class != class {
                            return Err(BindingError::WrongAddressSpace {
                                binding: class,
                                shader: self.class,
                            });
                        }
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
                BindingType::Sampler(ty) => {
                    if (ty == wgt::SamplerBindingType::Comparison) == comparison {
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
                            wgt::TextureSampleType::Depth => naga::ImageClass::Depth { multi },
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
                        let (naga_access, usage) = match access {
                            wgt::StorageTextureAccess::ReadOnly => (
                                naga::StorageAccess::LOAD,
                                GlobalUse::READ | GlobalUse::QUERY,
                            ),
                            wgt::StorageTextureAccess::WriteOnly => (
                                naga::StorageAccess::STORE,
                                GlobalUse::WRITE | GlobalUse::QUERY,
                            ),
                            wgt::StorageTextureAccess::ReadWrite => {
                                (naga::StorageAccess::all(), GlobalUse::all())
                            }
                        };
                        (
                            naga::ImageClass::Storage {
                                format: naga_format,
                                access: naga_access,
                            },
                            usage,
                        )
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

    fn derive_binding_type(
        &self,
        shader_usage: GlobalUse,
        features: wgt::Features,
    ) -> Result<BindingType, BindingError> {
        Ok(match self.ty {
            ResourceType::Buffer { size } => BindingType::Buffer {
                ty: match self.class {
                    naga::AddressSpace::Uniform => wgt::BufferBindingType::Uniform,
                    naga::AddressSpace::Storage { .. } => wgt::BufferBindingType::Storage {
                        read_only: !shader_usage.contains(GlobalUse::WRITE),
                    },
                    _ => return Err(BindingError::WrongType),
                },
                has_dynamic_offset: false,
                min_binding_size: Some(size),
            },
            ResourceType::Sampler { comparison } => BindingType::Sampler(if comparison {
                wgt::SamplerBindingType::Comparison
            } else {
                wgt::SamplerBindingType::Filtering
            }),
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
                    naga::ImageClass::Depth { multi } => BindingType::Texture {
                        sample_type: wgt::TextureSampleType::Depth,
                        view_dimension,
                        multisampled: multi,
                    },
                    naga::ImageClass::Storage { format, .. } => BindingType::StorageTexture {
                        access: if !shader_usage.contains(GlobalUse::READ) {
                            wgt::StorageTextureAccess::WriteOnly
                        } else if !features
                            .contains(wgt::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
                        {
                            return Err(BindingError::UnsupportedTextureStorageAccess(
                                shader_usage,
                            ));
                        } else if shader_usage.contains(GlobalUse::WRITE) {
                            wgt::StorageTextureAccess::ReadWrite
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
            Tf::R16Snorm | Tf::R16Unorm => (NumericDimension::Scalar, Sk::Float),
            Tf::Rg16Snorm | Tf::Rg16Unorm => (NumericDimension::Vector(Vs::Bi), Sk::Float),
            Tf::Rgba16Snorm | Tf::Rgba16Unorm => (NumericDimension::Vector(Vs::Quad), Sk::Float),
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
            Tf::Stencil8
            | Tf::Depth16Unorm
            | Tf::Depth32Float
            | Tf::Depth32FloatStencil8
            | Tf::Depth24Plus
            | Tf::Depth24PlusStencil8 => {
                panic!("Unexpected depth format")
            }
            Tf::Rgb9e5Ufloat => (NumericDimension::Vector(Vs::Tri), Sk::Float),
            Tf::Bc1RgbaUnorm
            | Tf::Bc1RgbaUnormSrgb
            | Tf::Bc2RgbaUnorm
            | Tf::Bc2RgbaUnormSrgb
            | Tf::Bc3RgbaUnorm
            | Tf::Bc3RgbaUnormSrgb
            | Tf::Bc7RgbaUnorm
            | Tf::Bc7RgbaUnormSrgb
            | Tf::Etc2Rgb8A1Unorm
            | Tf::Etc2Rgb8A1UnormSrgb
            | Tf::Etc2Rgba8Unorm
            | Tf::Etc2Rgba8UnormSrgb => (NumericDimension::Vector(Vs::Quad), Sk::Float),
            Tf::Bc4RUnorm | Tf::Bc4RSnorm | Tf::EacR11Unorm | Tf::EacR11Snorm => {
                (NumericDimension::Scalar, Sk::Float)
            }
            Tf::Bc5RgUnorm | Tf::Bc5RgSnorm | Tf::EacRg11Unorm | Tf::EacRg11Snorm => {
                (NumericDimension::Vector(Vs::Bi), Sk::Float)
            }
            Tf::Bc6hRgbUfloat | Tf::Bc6hRgbFloat | Tf::Etc2Rgb8Unorm | Tf::Etc2Rgb8UnormSrgb => {
                (NumericDimension::Vector(Vs::Tri), Sk::Float)
            }
            Tf::Astc {
                block: _,
                channel: _,
            } => (NumericDimension::Vector(Vs::Quad), Sk::Float),
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
        arena: &naga::UniqueArena<naga::Type>,
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
                //Note: technically this should be at least `log::error`, but
                // the reality is - every shader coming from `glslc` outputs an array
                // of clip distances and hits this path :(
                // So we lower it to `log::warn` to be less annoying.
                log::warn!("Unexpected varying type: {:?}", other);
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

    pub fn new(
        module: &naga::Module,
        info: &naga::valid::ModuleInfo,
        features: wgt::Features,
        limits: wgt::Limits,
    ) -> Self {
        let mut resources = naga::Arena::new();
        let mut resource_mapping = FastHashMap::default();
        for (var_handle, var) in module.global_variables.iter() {
            let bind = match var.binding {
                Some(ref br) => br.clone(),
                _ => continue,
            };
            let naga_ty = &module.types[var.ty].inner;

            let inner_ty = match *naga_ty {
                naga::TypeInner::BindingArray { base, .. } => &module.types[base].inner,
                ref ty => ty,
            };

            let ty = match *inner_ty {
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
                naga::TypeInner::Array { stride, .. } => ResourceType::Buffer {
                    size: wgt::BufferSize::new(stride as u64).unwrap(),
                },
                ref other => ResourceType::Buffer {
                    size: wgt::BufferSize::new(other.size(&module.constants) as u64).unwrap(),
                },
            };
            let handle = resources.append(
                Resource {
                    name: var.name.clone(),
                    bind,
                    ty,
                    class: var.space,
                },
                Default::default(),
            );
            resource_mapping.insert(var_handle, handle);
        }

        let mut entry_points = FastHashMap::default();
        entry_points.reserve(module.entry_points.len());
        for (index, entry_point) in module.entry_points.iter().enumerate() {
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

            for key in info.sampling_set.iter() {
                ep.sampling_pairs
                    .insert((resource_mapping[&key.image], resource_mapping[&key.sampler]));
            }

            ep.workgroup_size = entry_point.workgroup_size;

            entry_points.insert((entry_point.stage, entry_point.name.clone()), ep);
        }

        Self {
            features,
            limits,
            resources,
            entry_points,
        }
    }

    pub fn check_stage(
        &self,
        given_layouts: Option<&[&BindEntryMap]>,
        derived_layouts: &mut [BindEntryMap],
        shader_binding_sizes: &mut FastHashMap<naga::ResourceBinding, wgt::BufferSize>,
        entry_point_name: &str,
        stage_bit: wgt::ShaderStages,
        inputs: StageIo,
        compare_function: Option<wgt::CompareFunction>,
    ) -> Result<StageIo, StageError> {
        // Since a shader module can have multiple entry points with the same name,
        // we need to look for one with the right execution model.
        let shader_stage = match stage_bit {
            wgt::ShaderStages::VERTEX => naga::ShaderStage::Vertex,
            wgt::ShaderStages::FRAGMENT => naga::ShaderStage::Fragment,
            wgt::ShaderStages::COMPUTE => naga::ShaderStage::Compute,
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
                Some(layouts) => {
                    // update the required binding size for this buffer
                    if let ResourceType::Buffer { size } = res.ty {
                        match shader_binding_sizes.entry(res.bind.clone()) {
                            Entry::Occupied(e) => {
                                *e.into_mut() = size.max(*e.get());
                            }
                            Entry::Vacant(e) => {
                                e.insert(size);
                            }
                        }
                    }
                    layouts
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
                        .and_then(|entry| res.check_binding_use(entry, usage))
                }
                None => derived_layouts
                    .get_mut(res.bind.group as usize)
                    .ok_or(BindingError::Missing)
                    .and_then(|set| {
                        let ty = res.derive_binding_type(usage, self.features)?;
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

                let sampler_filtering = matches!(
                    sampler_layout.ty,
                    wgt::BindingType::Sampler(wgt::SamplerBindingType::Filtering)
                );
                let texture_sample_type = match texture_layout.ty {
                    BindingType::Texture { sample_type, .. } => sample_type,
                    _ => unreachable!(),
                };

                let error = match (sampler_filtering, texture_sample_type) {
                    (true, wgt::TextureSampleType::Float { filterable: false }) => {
                        Some(FilteringError::Float)
                    }
                    (true, wgt::TextureSampleType::Sint) => Some(FilteringError::Integer),
                    (true, wgt::TextureSampleType::Uint) => Some(FilteringError::Integer),
                    _ => None,
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

        // check workgroup size limits
        if shader_stage == naga::ShaderStage::Compute {
            let max_workgroup_size_limits = [
                self.limits.max_compute_workgroup_size_x,
                self.limits.max_compute_workgroup_size_y,
                self.limits.max_compute_workgroup_size_z,
            ];
            let total_invocations = entry_point.workgroup_size.iter().product::<u32>();

            if entry_point.workgroup_size.iter().any(|&s| s == 0)
                || total_invocations > self.limits.max_compute_invocations_per_workgroup
                || entry_point.workgroup_size[0] > max_workgroup_size_limits[0]
                || entry_point.workgroup_size[1] > max_workgroup_size_limits[1]
                || entry_point.workgroup_size[2] > max_workgroup_size_limits[2]
            {
                return Err(StageError::InvalidWorkgroupSize {
                    current: entry_point.workgroup_size,
                    current_total: total_invocations,
                    limit: max_workgroup_size_limits,
                    total: self.limits.max_compute_invocations_per_workgroup,
                });
            }
        }

        let mut inter_stage_components = 0;

        // check inputs compatibility
        for input in entry_point.inputs.iter() {
            match *input {
                Varying::Local { location, ref iv } => {
                    let result =
                        inputs
                            .get(&location)
                            .ok_or(InputError::Missing)
                            .and_then(|provided| {
                                let (compatible, num_components) = match shader_stage {
                                    // For vertex attributes, there are defaults filled out
                                    // by the driver if data is not provided.
                                    naga::ShaderStage::Vertex => {
                                        // vertex inputs don't count towards inter-stage
                                        (iv.ty.is_compatible_with(&provided.ty), 0)
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
                                        (
                                            iv.ty.is_subtype_of(&provided.ty),
                                            iv.ty.dim.num_components(),
                                        )
                                    }
                                    naga::ShaderStage::Compute => (false, 0),
                                };
                                if compatible {
                                    Ok(num_components)
                                } else {
                                    Err(InputError::WrongType(provided.ty))
                                }
                            });
                    match result {
                        Ok(num_components) => {
                            inter_stage_components += num_components;
                        }
                        Err(error) => {
                            return Err(StageError::Input {
                                location,
                                var: iv.clone(),
                                error,
                            })
                        }
                    }
                }
                Varying::BuiltIn(_) => {}
            }
        }

        // Check all vertex outputs and make sure the fragment shader consumes them.
        if shader_stage == naga::ShaderStage::Fragment {
            for &index in inputs.keys() {
                // This is a linear scan, but the count should be low enough
                // that this should be fine.
                let found = entry_point.inputs.iter().any(|v| match *v {
                    Varying::Local { location, .. } => location == index,
                    Varying::BuiltIn(_) => false,
                });

                if !found {
                    return Err(StageError::InputNotConsumed { location: index });
                }
            }
        }

        if shader_stage == naga::ShaderStage::Vertex {
            for output in entry_point.outputs.iter() {
                //TODO: count builtins towards the limit?
                inter_stage_components += match *output {
                    Varying::Local { ref iv, .. } => iv.ty.dim.num_components(),
                    Varying::BuiltIn(_) => 0,
                };

                if let Some(
                    cmp @ wgt::CompareFunction::Equal | cmp @ wgt::CompareFunction::NotEqual,
                ) = compare_function
                {
                    if let Varying::BuiltIn(naga::BuiltIn::Position { invariant: false }) = *output
                    {
                        log::warn!(
                            "Vertex shader with entry point {entry_point_name} outputs a @builtin(position) without the @invariant \
                            attribute and is used in a pipeline with {cmp:?}. On some machines, this can cause bad artifacting as {cmp:?} assumes \
                            the values output from the vertex shader exactly match the value in the depth buffer. The @invariant attribute on the \
                            @builtin(position) vertex output ensures that the exact same pixel depths are used every render."
                        );
                    }
                }
            }
        }

        if inter_stage_components > self.limits.max_inter_stage_shader_components {
            return Err(StageError::TooManyVaryings {
                used: inter_stage_components,
                limit: self.limits.max_inter_stage_shader_components,
            });
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

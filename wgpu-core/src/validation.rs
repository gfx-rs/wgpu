/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use crate::{binding_model::BindEntryMap, FastHashMap, MAX_BIND_GROUPS};
use arrayvec::ArrayVec;
use std::collections::hash_map::Entry;
use thiserror::Error;
use wgt::{BindGroupLayoutEntry, BindingType};

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
    #[error("load/store access flags {0:?} don't match the shader")]
    WrongUsage(naga::GlobalUse),
    #[error("type on the shader side does not match the pipeline binding")]
    WrongType,
    #[error("buffer structure size {0}, added to one element of an unbound array, if it's the last field, ended up greater than the given `min_binding_size`")]
    WrongBufferSize(wgt::BufferAddress),
    #[error("view dimension {dim:?} (is array: {is_array}) doesn't match the shader")]
    WrongTextureViewDimension {
        dim: naga::ImageDimension,
        is_array: bool,
    },
    #[error("component type {binding:?} of a sampled texture doesn't match the shader {shader:?}")]
    WrongTextureComponentType {
        binding: naga::ScalarKind,
        shader: naga::ScalarKind,
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
    #[error("texture format {0:?} isn't recognized")]
    UnknownStorageFormat(wgt::TextureFormat),
}

#[derive(Clone, Debug, Error)]
pub enum InputError {
    #[error("input is not provided by the earlier stage in the pipeline")]
    Missing,
    #[error("input type is not compatible with the provided")]
    WrongType,
}

/// Errors produced when validating a programmable stage of a pipeline.
#[derive(Clone, Debug, Error)]
pub enum StageError {
    #[error("shader module is invalid")]
    InvalidModule,
    #[error("unable to find an entry point matching the {0:?} execution model")]
    MissingEntryPoint(naga::ShaderStage),
    #[error("error matching global binding at index {binding} in set {set} against the pipeline layout: {error}")]
    Binding {
        set: u32,
        binding: u32,
        error: BindingError,
    },
    #[error(
        "error matching the stage input at {location} against the previous stage outputs: {error}"
    )]
    Input {
        location: wgt::ShaderLocation,
        error: InputError,
    },
}

fn get_aligned_type_size(
    module: &naga::Module,
    handle: naga::Handle<naga::Type>,
    allow_unbound: bool,
) -> wgt::BufferAddress {
    use naga::TypeInner as Ti;
    //TODO: take alignment into account!
    match module.types[handle].inner {
        Ti::Scalar { kind: _, width } => width as wgt::BufferAddress,
        Ti::Vector {
            size,
            kind: _,
            width,
        } => size as wgt::BufferAddress * width as wgt::BufferAddress,
        Ti::Matrix {
            rows,
            columns,
            kind: _,
            width,
        } => {
            rows as wgt::BufferAddress * columns as wgt::BufferAddress * width as wgt::BufferAddress
        }
        Ti::Pointer { .. } => 4,
        Ti::Array {
            base,
            size: naga::ArraySize::Static(count),
            stride,
        } => {
            let base_size = match stride {
                Some(stride) => stride.get() as wgt::BufferAddress,
                None => get_aligned_type_size(module, base, false),
            };
            base_size * count as wgt::BufferAddress
        }
        Ti::Array {
            base,
            size: naga::ArraySize::Dynamic,
            stride,
        } if allow_unbound => match stride {
            Some(stride) => stride.get() as wgt::BufferAddress,
            None => get_aligned_type_size(module, base, false),
        },
        Ti::Struct { ref members } => members.last().map_or(0, |member| {
            let offset = match member.origin {
                naga::MemberOrigin::BuiltIn(_) => {
                    tracing::error!("Missing offset on a struct member");
                    0 // TODO: make it a proper error
                }
                naga::MemberOrigin::Offset(offset) => offset as wgt::BufferAddress,
            };
            offset + get_aligned_type_size(module, member.ty, false)
        }),
        _ => panic!("Unexpected struct field"),
    }
}

fn check_binding_use(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    entry: &BindGroupLayoutEntry,
) -> Result<naga::GlobalUse, BindingError> {
    match module.types[var.ty].inner {
        naga::TypeInner::Struct { ref members } => {
            let (allowed_usage, min_size) = match entry.ty {
                BindingType::UniformBuffer {
                    dynamic: _,
                    min_binding_size,
                } => (naga::GlobalUse::LOAD, min_binding_size),
                BindingType::StorageBuffer {
                    dynamic: _,
                    min_binding_size,
                    readonly,
                } => {
                    let global_use = if readonly {
                        naga::GlobalUse::LOAD
                    } else {
                        naga::GlobalUse::all()
                    };
                    (global_use, min_binding_size)
                }
                _ => return Err(BindingError::WrongType),
            };
            let mut actual_size = 0;
            for (i, member) in members.iter().enumerate() {
                actual_size += get_aligned_type_size(module, member.ty, i + 1 == members.len());
            }
            match min_size {
                Some(non_zero) if non_zero.get() < actual_size => {
                    return Err(BindingError::WrongBufferSize(actual_size))
                }
                _ => (),
            }
            Ok(allowed_usage)
        }
        naga::TypeInner::Sampler { comparison } => match entry.ty {
            BindingType::Sampler { comparison: cmp } => {
                if cmp == comparison {
                    Ok(naga::GlobalUse::LOAD)
                } else {
                    Err(BindingError::WrongSamplerComparison)
                }
            }
            _ => Err(BindingError::WrongType),
        },
        naga::TypeInner::Image {
            kind,
            dim,
            arrayed,
            class,
        } => {
            let view_dimension = match entry.ty {
                BindingType::SampledTexture { dimension, .. }
                | BindingType::StorageTexture { dimension, .. } => dimension,
                _ => {
                    return Err(BindingError::WrongTextureViewDimension {
                        dim,
                        is_array: true,
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
                        })
                    }
                }
            }
            let expected_class = match entry.ty {
                BindingType::SampledTexture {
                    dimension: _,
                    component_type,
                    multisampled,
                } => {
                    let (expected_scalar_kind, comparison) = match component_type {
                        wgt::TextureComponentType::Float => (naga::ScalarKind::Float, false),
                        wgt::TextureComponentType::Sint => (naga::ScalarKind::Sint, false),
                        wgt::TextureComponentType::Uint => (naga::ScalarKind::Uint, false),
                        wgt::TextureComponentType::DepthComparison => {
                            (naga::ScalarKind::Float, true)
                        }
                    };
                    if kind != expected_scalar_kind {
                        return Err(BindingError::WrongTextureComponentType {
                            binding: expected_scalar_kind,
                            shader: kind,
                        });
                    }
                    if multisampled {
                        naga::ImageClass::Multisampled
                    } else if comparison {
                        naga::ImageClass::Depth
                    } else {
                        naga::ImageClass::Sampled
                    }
                }
                BindingType::StorageTexture {
                    readonly,
                    format,
                    dimension: _,
                } => {
                    let naga_format = match format {
                        wgt::TextureFormat::Rgba32Float => naga::StorageFormat::Rgba32f,
                        _ => return Err(BindingError::UnknownStorageFormat(format)),
                    };
                    let access = if readonly {
                        naga::StorageAccess::LOAD
                    } else {
                        naga::StorageAccess::STORE
                    };
                    naga::ImageClass::Storage(naga_format, access)
                }
                _ => return Err(BindingError::WrongType),
            };
            if class != expected_class {
                return Err(BindingError::WrongTextureClass {
                    binding: expected_class,
                    shader: class,
                });
            }
            Ok(match class {
                naga::ImageClass::Storage(_, access)
                    if access.contains(naga::StorageAccess::STORE) =>
                {
                    naga::GlobalUse::STORE
                }
                _ => naga::GlobalUse::LOAD,
            })
        }
        _ => Err(BindingError::WrongType),
    }
}

fn is_sub_type(sub: &naga::TypeInner, provided: &naga::TypeInner) -> bool {
    use naga::TypeInner as Ti;

    match (sub, provided) {
        (
            &Ti::Scalar {
                kind: k0,
                width: w0,
            },
            &Ti::Scalar {
                kind: k1,
                width: w1,
            },
        ) => k0 == k1 && w0 <= w1,
        (
            &Ti::Scalar {
                kind: k0,
                width: w0,
            },
            &Ti::Vector {
                size: _,
                kind: k1,
                width: w1,
            },
        ) => k0 == k1 && w0 <= w1,
        (
            &Ti::Vector {
                size: s0,
                kind: k0,
                width: w0,
            },
            &Ti::Vector {
                size: s1,
                kind: k1,
                width: w1,
            },
        ) => s0 as u8 <= s1 as u8 && k0 == k1 && w0 <= w1,
        (
            &Ti::Matrix {
                columns: c0,
                rows: r0,
                kind: k0,
                width: w0,
            },
            &Ti::Matrix {
                columns: c1,
                rows: r1,
                kind: k1,
                width: w1,
            },
        ) => c0 == c1 && r0 == r1 && k0 == k1 && w0 <= w1,
        (&Ti::Struct { members: ref m0 }, &Ti::Struct { members: ref m1 }) => m0 == m1,
        _ => false,
    }
}

pub enum MaybeOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> std::ops::Deref for MaybeOwned<'a, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match *self {
            MaybeOwned::Owned(ref value) => value,
            MaybeOwned::Borrowed(value) => value,
        }
    }
}

pub fn map_vertex_format(format: wgt::VertexFormat) -> naga::TypeInner {
    use naga::TypeInner as Ti;
    use wgt::VertexFormat as Vf;

    //Note: Shader always sees data as int, uint, or float.
    // It doesn't know if the original is normalized in a tighter form.
    let width = 4;
    match format {
        Vf::Uchar2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Uchar4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Char2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Char4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Uchar2Norm => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Uchar4Norm => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Char2Norm => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Char4Norm => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Ushort2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Ushort4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Short2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Short4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Ushort2Norm | Vf::Short2Norm | Vf::Half2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Ushort4Norm | Vf::Short4Norm | Vf::Half4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Float => Ti::Scalar {
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Float2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Float3 => Ti::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Float4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Float,
            width,
        },
        Vf::Uint => Ti::Scalar {
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Uint2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Uint3 => Ti::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Uint4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Uint,
            width,
        },
        Vf::Int => Ti::Scalar {
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Int2 => Ti::Vector {
            size: naga::VectorSize::Bi,
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Int3 => Ti::Vector {
            size: naga::VectorSize::Tri,
            kind: naga::ScalarKind::Sint,
            width,
        },
        Vf::Int4 => Ti::Vector {
            size: naga::VectorSize::Quad,
            kind: naga::ScalarKind::Sint,
            width,
        },
    }
}

fn map_texture_format(format: wgt::TextureFormat) -> naga::TypeInner {
    use naga::{ScalarKind as Sk, TypeInner as Ti, VectorSize as Vs};
    use wgt::TextureFormat as Tf;

    //Note: Shader always sees data as int, uint, or float.
    // It doesn't know if the original is normalized in a tighter form.
    let width = 4;
    match format {
        Tf::R8Unorm | Tf::R8Snorm => Ti::Scalar {
            kind: Sk::Float,
            width,
        },
        Tf::R8Uint => Ti::Scalar {
            kind: Sk::Uint,
            width,
        },
        Tf::R8Sint => Ti::Scalar {
            kind: Sk::Sint,
            width,
        },
        Tf::R16Uint => Ti::Scalar {
            kind: Sk::Uint,
            width,
        },
        Tf::R16Sint => Ti::Scalar {
            kind: Sk::Sint,
            width,
        },
        Tf::R16Float => Ti::Scalar {
            kind: Sk::Float,
            width,
        },
        Tf::Rg8Unorm | Tf::Rg8Snorm => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Float,
            width,
        },
        Tf::Rg8Uint => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Uint,
            width,
        },
        Tf::Rg8Sint => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Sint,
            width,
        },
        Tf::R32Uint => Ti::Scalar {
            kind: Sk::Uint,
            width,
        },
        Tf::R32Sint => Ti::Scalar {
            kind: Sk::Sint,
            width,
        },
        Tf::R32Float => Ti::Scalar {
            kind: Sk::Float,
            width,
        },
        Tf::Rg16Uint => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Uint,
            width,
        },
        Tf::Rg16Sint => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Sint,
            width,
        },
        Tf::Rg16Float => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Float,
            width,
        },
        Tf::Rgba8Unorm
        | Tf::Rgba8UnormSrgb
        | Tf::Rgba8Snorm
        | Tf::Bgra8Unorm
        | Tf::Bgra8UnormSrgb => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Float,
            width,
        },
        Tf::Rgba8Uint => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Uint,
            width,
        },
        Tf::Rgba8Sint => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Sint,
            width,
        },
        Tf::Rgb10a2Unorm => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Float,
            width,
        },
        Tf::Rg11b10Float => Ti::Vector {
            size: Vs::Tri,
            kind: Sk::Float,
            width,
        },
        Tf::Rg32Uint => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Uint,
            width,
        },
        Tf::Rg32Sint => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Sint,
            width,
        },
        Tf::Rg32Float => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Float,
            width,
        },
        Tf::Rgba16Uint => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Uint,
            width,
        },
        Tf::Rgba16Sint => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Sint,
            width,
        },
        Tf::Rgba16Float => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Float,
            width,
        },
        Tf::Rgba32Uint => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Uint,
            width,
        },
        Tf::Rgba32Sint => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Sint,
            width,
        },
        Tf::Rgba32Float => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Float,
            width,
        },
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
        | Tf::Bc7RgbaUnormSrgb => Ti::Vector {
            size: Vs::Quad,
            kind: Sk::Float,
            width,
        },
        Tf::Bc4RUnorm | Tf::Bc4RSnorm => Ti::Scalar {
            kind: Sk::Float,
            width,
        },
        Tf::Bc5RgUnorm | Tf::Bc5RgSnorm => Ti::Vector {
            size: Vs::Bi,
            kind: Sk::Float,
            width,
        },
        Tf::Bc6hRgbUfloat | Tf::Bc6hRgbSfloat => Ti::Vector {
            size: Vs::Tri,
            kind: Sk::Float,
            width,
        },
    }
}

/// Return true if the fragment `format` is covered by the provided `output`.
pub fn check_texture_format(format: wgt::TextureFormat, output: &naga::TypeInner) -> bool {
    let required = map_texture_format(format);
    is_sub_type(&required, output)
}

pub type StageInterface<'a> = FastHashMap<wgt::ShaderLocation, MaybeOwned<'a, naga::TypeInner>>;

pub enum IntrospectionBindGroupLayouts<'a> {
    Given(ArrayVec<[&'a BindEntryMap; MAX_BIND_GROUPS]>),
    Derived(&'a mut [BindEntryMap]),
}

fn derive_binding_type(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    usage: naga::GlobalUse,
) -> Result<BindingType, BindingError> {
    let ty = &module.types[var.ty];
    Ok(match ty.inner {
        naga::TypeInner::Struct { ref members } => {
            let dynamic = false;
            let mut actual_size = 0;
            for (i, member) in members.iter().enumerate() {
                actual_size += get_aligned_type_size(module, member.ty, i + 1 == members.len());
            }
            match var.class {
                naga::StorageClass::Uniform => BindingType::UniformBuffer {
                    dynamic,
                    min_binding_size: wgt::BufferSize::new(actual_size),
                },
                naga::StorageClass::StorageBuffer => BindingType::StorageBuffer {
                    dynamic,
                    min_binding_size: wgt::BufferSize::new(actual_size),
                    readonly: !usage.contains(naga::GlobalUse::STORE), //TODO: clarify
                },
                _ => return Err(BindingError::WrongType),
            }
        }
        naga::TypeInner::Sampler { comparison } => BindingType::Sampler { comparison },
        naga::TypeInner::Image {
            kind,
            dim,
            arrayed,
            class,
        } => {
            let dimension = match dim {
                naga::ImageDimension::D1 => wgt::TextureViewDimension::D1,
                naga::ImageDimension::D2 if arrayed => wgt::TextureViewDimension::D2Array,
                naga::ImageDimension::D2 => wgt::TextureViewDimension::D2,
                naga::ImageDimension::D3 => wgt::TextureViewDimension::D3,
                naga::ImageDimension::Cube if arrayed => wgt::TextureViewDimension::CubeArray,
                naga::ImageDimension::Cube => wgt::TextureViewDimension::Cube,
            };
            let component_type = match kind {
                naga::ScalarKind::Float => wgt::TextureComponentType::Float,
                naga::ScalarKind::Sint => wgt::TextureComponentType::Sint,
                naga::ScalarKind::Uint => wgt::TextureComponentType::Uint,
                naga::ScalarKind::Bool => unreachable!(),
            };
            match class {
                naga::ImageClass::Sampled => BindingType::SampledTexture {
                    dimension,
                    component_type,
                    multisampled: false,
                },
                naga::ImageClass::Multisampled => BindingType::SampledTexture {
                    dimension,
                    component_type,
                    multisampled: true,
                },
                naga::ImageClass::Depth => BindingType::SampledTexture {
                    dimension,
                    component_type: wgt::TextureComponentType::DepthComparison,
                    multisampled: false,
                },
                naga::ImageClass::Storage(format, access) => BindingType::StorageTexture {
                    dimension,
                    format: match format {
                        naga::StorageFormat::Rgba32f => wgt::TextureFormat::Rgba32Float,
                    },
                    readonly: !access.contains(naga::StorageAccess::STORE),
                },
            }
        }
        _ => return Err(BindingError::WrongType),
    })
}

pub fn check_stage<'a>(
    module: &'a naga::Module,
    mut group_layouts: IntrospectionBindGroupLayouts,
    entry_point_name: &str,
    stage: naga::ShaderStage,
    inputs: StageInterface<'a>,
) -> Result<StageInterface<'a>, StageError> {
    // Since a shader module can have multiple entry points with the same name,
    // we need to look for one with the right execution model.
    let entry_point = module
        .entry_points
        .iter()
        .find(|entry_point| entry_point.name == entry_point_name && entry_point.stage == stage)
        .ok_or(StageError::MissingEntryPoint(stage))?;
    let stage_bit = match stage {
        naga::ShaderStage::Vertex => wgt::ShaderStage::VERTEX,
        naga::ShaderStage::Fragment => wgt::ShaderStage::FRAGMENT,
        naga::ShaderStage::Compute => wgt::ShaderStage::COMPUTE,
    };

    let function = &module.functions[entry_point.function];
    let mut outputs = StageInterface::default();
    for ((_, var), &usage) in module.global_variables.iter().zip(&function.global_usage) {
        if usage.is_empty() {
            continue;
        }
        match var.binding {
            Some(naga::Binding::Descriptor { set, binding }) => {
                let result = match group_layouts {
                    IntrospectionBindGroupLayouts::Given(ref layouts) => layouts
                        .get(set as usize)
                        .and_then(|map| map.get(&binding))
                        .ok_or(BindingError::Missing)
                        .and_then(|entry| {
                            if entry.visibility.contains(stage_bit) {
                                Ok(entry)
                            } else {
                                Err(BindingError::Invisible)
                            }
                        })
                        .and_then(|entry| check_binding_use(module, var, entry))
                        .and_then(|allowed_usage| {
                            if allowed_usage.contains(usage) {
                                Ok(())
                            } else {
                                Err(BindingError::WrongUsage(usage))
                            }
                        }),
                    IntrospectionBindGroupLayouts::Derived(ref mut layouts) => layouts
                        .get_mut(set as usize)
                        .ok_or(BindingError::Missing)
                        .and_then(|set| {
                            let ty = derive_binding_type(module, var, usage)?;
                            Ok(match set.entry(binding) {
                                Entry::Occupied(e) if e.get().ty != ty => {
                                    return Err(BindingError::InconsistentlyDerivedType)
                                }
                                Entry::Occupied(e) => {
                                    e.into_mut().visibility |= stage_bit;
                                }
                                Entry::Vacant(e) => {
                                    e.insert(BindGroupLayoutEntry {
                                        binding,
                                        ty,
                                        visibility: stage_bit,
                                        count: None,
                                    });
                                }
                            })
                        }),
                };
                if let Err(error) = result {
                    return Err(StageError::Binding {
                        set,
                        binding,
                        error,
                    });
                }
            }
            Some(naga::Binding::Location(location)) => {
                let ty = &module.types[var.ty].inner;
                if usage.contains(naga::GlobalUse::STORE) {
                    outputs.insert(location, MaybeOwned::Borrowed(ty));
                } else {
                    let result =
                        inputs
                            .get(&location)
                            .ok_or(InputError::Missing)
                            .and_then(|provided| {
                                if is_sub_type(ty, provided) {
                                    Ok(())
                                } else {
                                    Err(InputError::WrongType)
                                }
                            });
                    if let Err(error) = result {
                        return Err(StageError::Input { location, error });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(outputs)
}

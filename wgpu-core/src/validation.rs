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
    #[error("unable to find an entry point at {0:?} stage")]
    MissingEntryPoint(wgt::ShaderStage),
    #[error("error matching global binding at index {binding} in group {group} against the pipeline layout: {error}")]
    Binding {
        group: u32,
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
            width,
        } => {
            rows as wgt::BufferAddress * columns as wgt::BufferAddress * width as wgt::BufferAddress
        }
        Ti::Pointer { .. } => 4,
        Ti::Array {
            base,
            size: naga::ArraySize::Constant(const_handle),
            stride,
        } => {
            let base_size = match stride {
                Some(stride) => stride.get() as wgt::BufferAddress,
                None => get_aligned_type_size(module, base, false),
            };
            let count = match module.constants[const_handle].inner {
                naga::ConstantInner::Uint(value) => value,
                ref other => panic!("Invalid array size constant: {:?}", other),
            };
            base_size * count
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
                naga::MemberOrigin::Empty => 0,
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

fn check_binding_use(
    module: &naga::Module,
    var: &naga::GlobalVariable,
    entry: &BindGroupLayoutEntry,
) -> Result<naga::GlobalUse, BindingError> {
    match module.types[var.ty].inner {
        naga::TypeInner::Struct { ref members } => {
            let (allowed_usage, min_size) = match entry.ty {
                BindingType::Buffer {
                    ty,
                    has_dynamic_offset: _,
                    min_binding_size,
                } => {
                    let global_use = match ty {
                        wgt::BufferBindingType::Uniform
                        | wgt::BufferBindingType::Storage { read_only: true } => {
                            naga::GlobalUse::LOAD
                        }
                        wgt::BufferBindingType::Storage { read_only: _ } => naga::GlobalUse::all(),
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
            BindingType::Sampler {
                filtering: _,
                comparison: cmp,
            } => {
                if cmp == comparison {
                    Ok(naga::GlobalUse::LOAD)
                } else {
                    Err(BindingError::WrongSamplerComparison)
                }
            }
            _ => Err(BindingError::WrongType),
        },
        naga::TypeInner::Image {
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
                    (class, naga::GlobalUse::LOAD)
                }
                BindingType::StorageTexture {
                    access,
                    format,
                    view_dimension: _,
                } => {
                    let naga_format = map_storage_format_to_naga(format)
                        .ok_or(BindingError::BadStorageFormat(format))?;
                    let usage = match access {
                        wgt::StorageTextureAccess::ReadOnly => naga::GlobalUse::LOAD,
                        wgt::StorageTextureAccess::WriteOnly => naga::GlobalUse::STORE,
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
            Ok(usage)
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
                width: w0,
            },
            &Ti::Matrix {
                columns: c1,
                rows: r1,
                width: w1,
            },
        ) => c0 == c1 && r0 == r1 && w0 <= w1,
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
            let has_dynamic_offset = false;
            let mut actual_size = 0;
            for (i, member) in members.iter().enumerate() {
                actual_size += get_aligned_type_size(module, member.ty, i + 1 == members.len());
            }
            match var.class {
                naga::StorageClass::Uniform => BindingType::Buffer {
                    ty: wgt::BufferBindingType::Uniform,
                    has_dynamic_offset,
                    min_binding_size: wgt::BufferSize::new(actual_size),
                },
                naga::StorageClass::Storage => BindingType::Buffer {
                    ty: wgt::BufferBindingType::Storage {
                        read_only: !usage.contains(naga::GlobalUse::STORE),
                    },
                    has_dynamic_offset,
                    min_binding_size: wgt::BufferSize::new(actual_size),
                },
                _ => return Err(BindingError::WrongType),
            }
        }
        naga::TypeInner::Sampler { comparison } => BindingType::Sampler {
            filtering: true,
            comparison,
        },
        naga::TypeInner::Image {
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
                    access: if usage.contains(naga::GlobalUse::STORE) {
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
        _ => return Err(BindingError::WrongType),
    })
}

pub fn check_stage<'a>(
    module: &'a naga::Module,
    mut group_layouts: IntrospectionBindGroupLayouts,
    entry_point_name: &str,
    stage_bit: wgt::ShaderStage,
    inputs: StageInterface<'a>,
) -> Result<StageInterface<'a>, StageError> {
    // Since a shader module can have multiple entry points with the same name,
    // we need to look for one with the right execution model.
    let shader_stage = match stage_bit {
        wgt::ShaderStage::VERTEX => naga::ShaderStage::Vertex,
        wgt::ShaderStage::FRAGMENT => naga::ShaderStage::Fragment,
        wgt::ShaderStage::COMPUTE => naga::ShaderStage::Compute,
        _ => unreachable!(),
    };
    let entry_point = module
        .entry_points
        .get(&(shader_stage, entry_point_name.to_string()))
        .ok_or(StageError::MissingEntryPoint(stage_bit))?;

    let mut outputs = StageInterface::default();
    for ((_, var), &usage) in module
        .global_variables
        .iter()
        .zip(&entry_point.function.global_usage)
    {
        if usage.is_empty() {
            continue;
        }
        match var.binding {
            Some(naga::Binding::Resource { group, binding }) => {
                let result = match group_layouts {
                    IntrospectionBindGroupLayouts::Given(ref layouts) => layouts
                        .get(group as usize)
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
                        .get_mut(group as usize)
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
                        group,
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

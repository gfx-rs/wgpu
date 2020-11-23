use super::{
    error::{BackendResult, Error},
    Version, Writer,
};
use crate::{
    Bytes, ImageClass, ImageDimension, ScalarKind, ShaderStage, StorageClass, StorageFormat,
    TypeInner,
};
use std::io::Write;

bitflags::bitflags! {
    pub struct Features: u32 {
        const BUFFER_STORAGE = 1;
        const ARRAY_OF_ARRAYS = 1 << 1;
        const DOUBLE_TYPE = 1 << 2;
        const FULL_IMAGE_FORMATS = 1 << 3;
        const MULTISAMPLED_TEXTURES = 1 << 4;
        const MULTISAMPLED_TEXTURE_ARRAYS = 1 << 5;
        const CUBE_TEXTURES_ARRAY = 1 << 6;
        const COMPUTE_SHADER = 1 << 7;
        const IMAGE_LOAD_STORE = 1 << 8;
        const CONSERVATIVE_DEPTH = 1 << 9;
        const TEXTURE_1D = 1 << 10;
        const PUSH_CONSTANT = 1 << 11;
    }
}

pub struct FeaturesManager(Features);

impl FeaturesManager {
    pub fn new() -> Self {
        Self(Features::empty())
    }

    pub fn request(&mut self, features: Features) {
        self.0 |= features
    }

    #[allow(clippy::collapsible_if)]
    pub fn write(&self, version: Version, mut out: impl Write) -> BackendResult {
        if self.0.contains(Features::COMPUTE_SHADER) {
            if version < Version::Embedded(310) || version < Version::Desktop(420) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support compute shaders",
                    version
                )));
            }

            if !version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_compute_shader.txt
                writeln!(out, "#extension GL_ARB_compute_shader : require")?;
            }
        }

        if self.0.contains(Features::BUFFER_STORAGE) {
            if version < Version::Embedded(310) || version < Version::Desktop(400) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support buffer storage class",
                    version
                )));
            }

            if let Version::Desktop(_) = version {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_storage_buffer_object.txt
                writeln!(
                    out,
                    "#extension GL_ARB_shader_storage_buffer_object : require"
                )?;
            }
        }

        if self.0.contains(Features::DOUBLE_TYPE) {
            if version.is_es() || version < Version::Desktop(150) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support doubles",
                    version
                )));
            }

            if version < Version::Desktop(400) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_gpu_shader_fp64.txt
                writeln!(out, "#extension GL_ARB_gpu_shader_fp64 : require")?;
            }
        }

        if self.0.contains(Features::CUBE_TEXTURES_ARRAY) {
            if version < Version::Embedded(310) || version < Version::Desktop(130) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support cube map array textures",
                    version
                )));
            }

            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_cube_map_array.txt
                writeln!(out, "#extension GL_EXT_texture_cube_map_array : require")?;
            } else if version < Version::Desktop(400) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_texture_cube_map_array.txt
                writeln!(out, "#extension GL_ARB_texture_cube_map_array : require")?;
            }
        }

        if self.0.contains(Features::MULTISAMPLED_TEXTURES) {
            if version < Version::Embedded(300) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support multi sampled textures",
                    version
                )));
            }
        }

        if self.0.contains(Features::MULTISAMPLED_TEXTURE_ARRAYS) {
            if version < Version::Embedded(310) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support multi sampled texture arrays",
                    version
                )));
            }

            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/OES/OES_texture_storage_multisample_2d_array.txt
                writeln!(
                    out,
                    "#extension GL_OES_texture_storage_multisample_2d_array : require"
                )?;
            }
        }

        if self.0.contains(Features::ARRAY_OF_ARRAYS) {
            if version < Version::Embedded(310) || version < Version::Desktop(120) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't arrays of arrays",
                    version
                )));
            }

            if version < Version::Desktop(430) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_arrays_of_arrays.txt
                writeln!(out, "#extension ARB_arrays_of_arrays : require")?;
            }
        }

        if self.0.contains(Features::IMAGE_LOAD_STORE) {
            if version < Version::Embedded(310) || version < Version::Desktop(130) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support images load/stores",
                    version
                )));
            }

            if self.0.contains(Features::FULL_IMAGE_FORMATS) && version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/NV/NV_image_formats.txt
                writeln!(out, "#extension GL_NV_image_formats : require")?;
            }

            if version < Version::Desktop(420) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_image_load_store.txt
                writeln!(out, "#extension GL_ARB_shader_image_load_store : require")?;
            }
        }

        if self.0.contains(Features::CONSERVATIVE_DEPTH) {
            if version < Version::Embedded(300) || version < Version::Desktop(130) {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support conservative depth",
                    version
                )));
            }

            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_conservative_depth.txt
                writeln!(out, "#extension GL_EXT_conservative_depth : require")?;
            }

            if version < Version::Desktop(420) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_conservative_depth.txt
                writeln!(out, "#extension GL_ARB_conservative_depth : require")?;
            }
        }

        if self.0.contains(Features::TEXTURE_1D) {
            if version.is_es() {
                return Err(Error::Custom(format!(
                    "Version {} doesn't support 1d textures",
                    version
                )));
            }
        }

        Ok(())
    }
}

impl<'a, W> Writer<'a, W> {
    pub(super) fn collect_required_features(&mut self) {
        let stage = self.options.entry_point.0;

        if let Some(depth_test) = self.entry_point.early_depth_test {
            self.features.request(Features::IMAGE_LOAD_STORE);

            if depth_test.conservative.is_some() {
                self.features.request(Features::CONSERVATIVE_DEPTH);
            }
        }

        if let ShaderStage::Compute = stage {
            self.features.request(Features::COMPUTE_SHADER)
        }

        for (_, ty) in self.module.types.iter() {
            match ty.inner {
                TypeInner::Scalar { kind, width } => self.scalar_required_features(kind, width),
                TypeInner::Vector { kind, width, .. } => self.scalar_required_features(kind, width),
                TypeInner::Matrix { .. } => self.scalar_required_features(ScalarKind::Float, 8),
                TypeInner::Array { base, .. } => {
                    if let TypeInner::Array { .. } = self.module.types[base].inner {
                        self.features.request(Features::ARRAY_OF_ARRAYS)
                    }
                }
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    if arrayed && dim == ImageDimension::Cube {
                        self.features.request(Features::CUBE_TEXTURES_ARRAY)
                    } else if dim == ImageDimension::D1 {
                        self.features.request(Features::TEXTURE_1D)
                    }

                    match class {
                        ImageClass::Sampled { multi: true, .. } => {
                            self.features.request(Features::MULTISAMPLED_TEXTURES);
                            if arrayed {
                                self.features.request(Features::MULTISAMPLED_TEXTURE_ARRAYS);
                            }
                        }
                        ImageClass::Storage(format) => match format {
                            StorageFormat::R8Unorm
                            | StorageFormat::R8Snorm
                            | StorageFormat::R8Uint
                            | StorageFormat::R8Sint
                            | StorageFormat::R16Uint
                            | StorageFormat::R16Sint
                            | StorageFormat::R16Float
                            | StorageFormat::Rg8Unorm
                            | StorageFormat::Rg8Snorm
                            | StorageFormat::Rg8Uint
                            | StorageFormat::Rg8Sint
                            | StorageFormat::Rg16Uint
                            | StorageFormat::Rg16Sint
                            | StorageFormat::Rg16Float
                            | StorageFormat::Rgb10a2Unorm
                            | StorageFormat::Rg11b10Float
                            | StorageFormat::Rg32Uint
                            | StorageFormat::Rg32Sint
                            | StorageFormat::Rg32Float => {
                                self.features.request(Features::FULL_IMAGE_FORMATS)
                            }
                            _ => {}
                        },
                        _ => {}
                    }
                }
                _ => {}
            }
        }

        for (_, global) in self.module.global_variables.iter() {
            match global.class {
                StorageClass::WorkGroup => self.features.request(Features::COMPUTE_SHADER),
                StorageClass::Storage => self.features.request(Features::BUFFER_STORAGE),
                StorageClass::PushConstant => self.features.request(Features::PUSH_CONSTANT),
                _ => {}
            }
        }
    }

    fn scalar_required_features(&mut self, kind: ScalarKind, width: Bytes) {
        if kind == ScalarKind::Float && width == 8 {
            self.features.request(Features::DOUBLE_TYPE);
        }
    }
}

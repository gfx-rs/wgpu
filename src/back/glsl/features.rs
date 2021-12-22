use super::{BackendResult, Error, Version, Writer};
use crate::{
    Binding, Bytes, Expression, Handle, ImageClass, ImageDimension, Interpolation, MathFunction,
    Sampling, ScalarKind, ShaderStage, StorageClass, StorageFormat, Type, TypeInner,
};
use std::fmt::Write;

bitflags::bitflags! {
    /// Structure used to encode a set of additions to glsl that aren't supported by all versions
    pub struct Features: u32 {
        /// Buffer storage class support
        const BUFFER_STORAGE = 1;
        const ARRAY_OF_ARRAYS = 1 << 1;
        /// 8 byte floats
        const DOUBLE_TYPE = 1 << 2;
        /// Includes support for more image formats
        const FULL_IMAGE_FORMATS = 1 << 3;
        const MULTISAMPLED_TEXTURES = 1 << 4;
        const MULTISAMPLED_TEXTURE_ARRAYS = 1 << 5;
        const CUBE_TEXTURES_ARRAY = 1 << 6;
        const COMPUTE_SHADER = 1 << 7;
        /// Adds support for image load and early depth tests
        const IMAGE_LOAD_STORE = 1 << 8;
        const CONSERVATIVE_DEPTH = 1 << 9;
        /// Interpolation and auxiliary qualifiers. Perspective, Flat, and
        /// Centroid are available in all GLSL versions we support.
        const NOPERSPECTIVE_QUALIFIER = 1 << 11;
        const SAMPLE_QUALIFIER = 1 << 12;
        const CLIP_DISTANCE = 1 << 13;
        const CULL_DISTANCE = 1 << 14;
        // Sample ID
        const SAMPLE_VARIABLES = 1 << 15;
        /// Arrays with a dynamic length
        const DYNAMIC_ARRAY_SIZE = 1 << 16;
        const MULTI_VIEW = 1 << 17;
        /// Adds support for fused multiply-add
        const FMA = 1 << 18;
    }
}

/// Helper structure used to store the required [`Features`](Features) needed to output a
/// [`Module`](crate::Module)
///
/// Provides helper methods to check for availability and writing required extensions
pub struct FeaturesManager(Features);

impl FeaturesManager {
    /// Creates a new [`FeaturesManager`](FeaturesManager) instance
    pub fn new() -> Self {
        Self(Features::empty())
    }

    /// Adds to the list of required [`Features`](Features)
    pub fn request(&mut self, features: Features) {
        self.0 |= features
    }

    /// Checks that all required [`Features`](Features) are available for the specified
    /// [`Version`](super::Version) otherwise returns an
    /// [`Error::MissingFeatures`](super::Error::MissingFeatures)
    pub fn check_availability(&self, version: Version) -> BackendResult {
        // Will store all the features that are unavailable
        let mut missing = Features::empty();

        // Helper macro to check for feature availability
        macro_rules! check_feature {
            // Used when only core glsl supports the feature
            ($feature:ident, $core:literal) => {
                if self.0.contains(Features::$feature)
                    && (version < Version::Desktop($core) || version.is_es())
                {
                    missing |= Features::$feature;
                }
            };
            // Used when both core and es support the feature
            ($feature:ident, $core:literal, $es:literal) => {
                if self.0.contains(Features::$feature)
                    && (version < Version::Desktop($core) || version < Version::Embedded($es))
                {
                    missing |= Features::$feature;
                }
            };
        }

        check_feature!(COMPUTE_SHADER, 420, 310);
        check_feature!(BUFFER_STORAGE, 400, 310);
        check_feature!(DOUBLE_TYPE, 150);
        check_feature!(CUBE_TEXTURES_ARRAY, 130, 310);
        check_feature!(MULTISAMPLED_TEXTURES, 150, 300);
        check_feature!(MULTISAMPLED_TEXTURE_ARRAYS, 150, 310);
        check_feature!(ARRAY_OF_ARRAYS, 120, 310);
        check_feature!(IMAGE_LOAD_STORE, 130, 310);
        check_feature!(CONSERVATIVE_DEPTH, 130, 300);
        check_feature!(CONSERVATIVE_DEPTH, 130, 300);
        check_feature!(NOPERSPECTIVE_QUALIFIER, 130);
        check_feature!(SAMPLE_QUALIFIER, 400, 320);
        // gl_ClipDistance is supported by core versions > 1.3 and aren't supported by an es versions without extensions
        check_feature!(CLIP_DISTANCE, 130, 300);
        check_feature!(CULL_DISTANCE, 450, 300);
        check_feature!(SAMPLE_VARIABLES, 400, 300);
        check_feature!(DYNAMIC_ARRAY_SIZE, 430, 310);
        check_feature!(MULTI_VIEW, 140, 310);
        check_feature!(FMA, 400, 310);

        // Return an error if there are missing features
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::MissingFeatures(missing))
        }
    }

    /// Helper method used to write all needed extensions
    ///
    /// # Notes
    /// This won't check for feature availability so it might output extensions that aren't even
    /// supported.[`check_availability`](Self::check_availability) will check feature availability
    pub fn write(&self, version: Version, mut out: impl Write) -> BackendResult {
        if self.0.contains(Features::COMPUTE_SHADER) && !version.is_es() {
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_compute_shader.txt
            writeln!(out, "#extension GL_ARB_compute_shader : require")?;
        }

        if self.0.contains(Features::BUFFER_STORAGE) && !version.is_es() {
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_shader_storage_buffer_object.txt
            writeln!(
                out,
                "#extension GL_ARB_shader_storage_buffer_object : require"
            )?;
        }

        if self.0.contains(Features::DOUBLE_TYPE) && version < Version::Desktop(400) {
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_gpu_shader_fp64.txt
            writeln!(out, "#extension GL_ARB_gpu_shader_fp64 : require")?;
        }

        if self.0.contains(Features::CUBE_TEXTURES_ARRAY) {
            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_texture_cube_map_array.txt
                writeln!(out, "#extension GL_EXT_texture_cube_map_array : require")?;
            } else if version < Version::Desktop(400) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_texture_cube_map_array.txt
                writeln!(out, "#extension GL_ARB_texture_cube_map_array : require")?;
            }
        }

        if self.0.contains(Features::MULTISAMPLED_TEXTURE_ARRAYS) && version.is_es() {
            // https://www.khronos.org/registry/OpenGL/extensions/OES/OES_texture_storage_multisample_2d_array.txt
            writeln!(
                out,
                "#extension GL_OES_texture_storage_multisample_2d_array : require"
            )?;
        }

        if self.0.contains(Features::ARRAY_OF_ARRAYS) && version < Version::Desktop(430) {
            // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_arrays_of_arrays.txt
            writeln!(out, "#extension ARB_arrays_of_arrays : require")?;
        }

        if self.0.contains(Features::IMAGE_LOAD_STORE) {
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
            if version.is_es() {
                // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_conservative_depth.txt
                writeln!(out, "#extension GL_EXT_conservative_depth : require")?;
            }

            if version < Version::Desktop(420) {
                // https://www.khronos.org/registry/OpenGL/extensions/ARB/ARB_conservative_depth.txt
                writeln!(out, "#extension GL_ARB_conservative_depth : require")?;
            }
        }

        if (self.0.contains(Features::CLIP_DISTANCE) || self.0.contains(Features::CULL_DISTANCE))
            && version.is_es()
        {
            // TODO: handle gl_ClipDistance and gl_CullDistance usage in better way
            // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_clip_cull_distance.txt
            // writeln!(out, "#extension GL_EXT_clip_cull_distance : require")?;
        }

        if self.0.contains(Features::SAMPLE_VARIABLES) && version.is_es() {
            // https://www.khronos.org/registry/OpenGL/extensions/OES/OES_sample_variables.txt
            writeln!(out, "#extension GL_OES_sample_variables : require")?;
        }

        if self.0.contains(Features::SAMPLE_VARIABLES) && version.is_es() {
            // https://www.khronos.org/registry/OpenGL/extensions/OES/OES_sample_variables.txt
            writeln!(out, "#extension GL_OES_sample_variables : require")?;
        }

        if self.0.contains(Features::MULTI_VIEW) {
            // https://github.com/KhronosGroup/GLSL/blob/master/extensions/ext/GL_EXT_multiview.txt
            writeln!(out, "#extension GL_EXT_multiview : require")?;
        }

        if self.0.contains(Features::FMA) && version.is_es() {
            // https://www.khronos.org/registry/OpenGL/extensions/EXT/EXT_gpu_shader5.txt
            writeln!(out, "#extension GL_EXT_gpu_shader5 : require")?;
        }

        Ok(())
    }
}

impl<'a, W> Writer<'a, W> {
    /// Helper method that searches the module for all the needed [`Features`](Features)
    ///
    /// # Errors
    /// If the version doesn't support any of the needed [`Features`](Features) a
    /// [`Error::MissingFeatures`](super::Error::MissingFeatures) will be returned
    pub(super) fn collect_required_features(&mut self) -> BackendResult {
        let ep_info = self.info.get_entry_point(self.entry_point_idx as usize);

        if let Some(depth_test) = self.entry_point.early_depth_test {
            // If IMAGE_LOAD_STORE is supported for this version of GLSL
            if self.options.version.supports_early_depth_test() {
                self.features.request(Features::IMAGE_LOAD_STORE);
            }

            if depth_test.conservative.is_some() {
                self.features.request(Features::CONSERVATIVE_DEPTH);
            }
        }

        for arg in self.entry_point.function.arguments.iter() {
            self.varying_required_features(arg.binding.as_ref(), arg.ty);
        }
        if let Some(ref result) = self.entry_point.function.result {
            self.varying_required_features(result.binding.as_ref(), result.ty);
        }

        if let ShaderStage::Compute = self.entry_point.stage {
            self.features.request(Features::COMPUTE_SHADER)
        }

        for (ty_handle, ty) in self.module.types.iter() {
            match ty.inner {
                TypeInner::Scalar { kind, width } => self.scalar_required_features(kind, width),
                TypeInner::Vector { kind, width, .. } => self.scalar_required_features(kind, width),
                TypeInner::Matrix { width, .. } => {
                    self.scalar_required_features(ScalarKind::Float, width)
                }
                TypeInner::Array { base, size, .. } => {
                    if let TypeInner::Array { .. } = self.module.types[base].inner {
                        self.features.request(Features::ARRAY_OF_ARRAYS)
                    }

                    // If the array is dynamically sized
                    if size == crate::ArraySize::Dynamic {
                        let mut is_used = false;

                        // Check if this type is used in a global that is needed by the current entrypoint
                        for (global_handle, global) in self.module.global_variables.iter() {
                            // Skip unused globals
                            if ep_info[global_handle].is_empty() {
                                continue;
                            }

                            // If this array is the type of a global, then this array is used
                            if global.ty == ty_handle {
                                is_used = true;
                                break;
                            }

                            // If the type of this global is a struct
                            if let crate::TypeInner::Struct { ref members, .. } =
                                self.module.types[global.ty].inner
                            {
                                // Check the last element of the struct to see if it's type uses
                                // this array
                                if let Some(last) = members.last() {
                                    if last.ty == ty_handle {
                                        is_used = true;
                                        break;
                                    }
                                }
                            }
                        }

                        // If this dynamically size array is used, we need dynamic array size support
                        if is_used {
                            self.features.request(Features::DYNAMIC_ARRAY_SIZE);
                        }
                    }
                }
                TypeInner::Image {
                    dim,
                    arrayed,
                    class,
                } => {
                    if arrayed && dim == ImageDimension::Cube {
                        self.features.request(Features::CUBE_TEXTURES_ARRAY)
                    }

                    match class {
                        ImageClass::Sampled { multi: true, .. }
                        | ImageClass::Depth { multi: true } => {
                            self.features.request(Features::MULTISAMPLED_TEXTURES);
                            if arrayed {
                                self.features.request(Features::MULTISAMPLED_TEXTURE_ARRAYS);
                            }
                        }
                        ImageClass::Storage { format, .. } => match format {
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
                        ImageClass::Sampled { multi: false, .. }
                        | ImageClass::Depth { multi: false } => {}
                    }
                }
                _ => {}
            }
        }

        for (handle, global) in self.module.global_variables.iter() {
            if ep_info[handle].is_empty() {
                continue;
            }
            match global.class {
                StorageClass::WorkGroup => self.features.request(Features::COMPUTE_SHADER),
                StorageClass::Storage { .. } => self.features.request(Features::BUFFER_STORAGE),
                StorageClass::PushConstant => return Err(Error::PushConstantNotSupported),
                _ => {}
            }
        }

        if self.options.version.supports_fma_function() {
            let has_fma = self
                .module
                .functions
                .iter()
                .flat_map(|(_, f)| f.expressions.iter())
                .chain(
                    self.module
                        .entry_points
                        .iter()
                        .flat_map(|e| e.function.expressions.iter()),
                )
                .any(|(_, e)| match *e {
                    Expression::Math { fun, .. } if fun == MathFunction::Fma => true,
                    _ => false,
                });
            if has_fma {
                self.features.request(Features::FMA);
            }
        }

        self.features.check_availability(self.options.version)
    }

    /// Helper method that checks the [`Features`](Features) needed by a scalar
    fn scalar_required_features(&mut self, kind: ScalarKind, width: Bytes) {
        if kind == ScalarKind::Float && width == 8 {
            self.features.request(Features::DOUBLE_TYPE);
        }
    }

    fn varying_required_features(&mut self, binding: Option<&Binding>, ty: Handle<Type>) {
        match self.module.types[ty].inner {
            crate::TypeInner::Struct { ref members, .. } => {
                for member in members {
                    self.varying_required_features(member.binding.as_ref(), member.ty);
                }
            }
            _ => {
                if let Some(binding) = binding {
                    match *binding {
                        Binding::BuiltIn(builtin) => match builtin {
                            crate::BuiltIn::ClipDistance => {
                                self.features.request(Features::CLIP_DISTANCE)
                            }
                            crate::BuiltIn::CullDistance => {
                                self.features.request(Features::CULL_DISTANCE)
                            }
                            crate::BuiltIn::SampleIndex => {
                                self.features.request(Features::SAMPLE_VARIABLES)
                            }
                            crate::BuiltIn::ViewIndex => {
                                self.features.request(Features::MULTI_VIEW)
                            }
                            _ => {}
                        },
                        Binding::Location {
                            location: _,
                            interpolation,
                            sampling,
                        } => {
                            if interpolation == Some(Interpolation::Linear) {
                                self.features.request(Features::NOPERSPECTIVE_QUALIFIER);
                            }
                            if sampling == Some(Sampling::Sample) {
                                self.features.request(Features::SAMPLE_QUALIFIER);
                            }
                        }
                    }
                }
            }
        }
    }
}

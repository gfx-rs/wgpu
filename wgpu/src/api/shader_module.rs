use core::{future::Future, marker::PhantomData};
pub use wgt::{CompilationInfo, CompilationMessage, CompilationMessageType, SourceLocation};

use crate::*;

/// Handle to a compiled shader module.
///
/// A `ShaderModule` represents a compiled shader module on the GPU. It can be created by passing
/// source code to [`Device::create_shader_module`]. MSL shader or SPIR-V binary can also be passed
/// directly using [`Device::create_shader_module_passthrough`]. Shader modules are used to define
/// programmable stages of a pipeline.
///
/// Corresponds to [WebGPU `GPUShaderModule`](https://gpuweb.github.io/gpuweb/#shader-module).
#[derive(Debug, Clone)]
pub struct ShaderModule {
    pub(crate) inner: dispatch::DispatchShaderModule,
}
#[cfg(send_sync)]
static_assertions::assert_impl_all!(ShaderModule: Send, Sync);

crate::cmp::impl_eq_ord_hash_proxy!(ShaderModule => .inner);

impl ShaderModule {
    /// Get the compilation info for the shader module.
    pub fn get_compilation_info(&self) -> impl Future<Output = CompilationInfo> + WasmNotSend {
        self.inner.get_compilation_info()
    }

    #[cfg(custom)]
    /// Returns custom implementation of ShaderModule (if custom backend and is internally T)
    pub fn as_custom<T: custom::ShaderModuleInterface>(&self) -> Option<&T> {
        self.inner.as_custom()
    }
}

/// Source of a shader module.
///
/// The source will be parsed and validated.
///
/// Any necessary shader translation (e.g. from WGSL to SPIR-V or vice versa)
/// will be done internally by wgpu.
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// only WGSL source code strings are accepted.
#[cfg_attr(feature = "naga-ir", expect(clippy::large_enum_variant))]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum ShaderSource<'a> {
    /// SPIR-V module represented as a slice of words.
    ///
    /// See also: [`util::make_spirv`], [`include_spirv`]
    #[cfg(feature = "spirv")]
    SpirV(alloc::borrow::Cow<'a, [u32]>),
    /// GLSL module as a string slice.
    ///
    /// Note: GLSL is not yet fully supported and must be a specific ShaderStage.
    #[cfg(feature = "glsl")]
    Glsl {
        /// The source code of the shader.
        shader: alloc::borrow::Cow<'a, str>,
        /// The shader stage that the shader targets. For example, `naga::ShaderStage::Vertex`
        stage: naga::ShaderStage,
        /// Key-value pairs to represent defines sent to the glsl preprocessor.
        ///
        /// If the same name is defined multiple times, the last value is used.
        defines: &'a [(&'a str, &'a str)],
    },
    /// WGSL module as a string slice.
    #[cfg(feature = "wgsl")]
    Wgsl(alloc::borrow::Cow<'a, str>),
    /// Naga module.
    #[cfg(feature = "naga-ir")]
    Naga(alloc::borrow::Cow<'static, naga::Module>),
    /// Dummy variant because `Naga` doesn't have a lifetime and without enough active features it
    /// could be the last one active.
    #[doc(hidden)]
    Dummy(PhantomData<&'a ()>),
}
static_assertions::assert_impl_all!(ShaderSource<'_>: Send, Sync);

/// Descriptor for use with [`Device::create_shader_module`].
///
/// Corresponds to [WebGPU `GPUShaderModuleDescriptor`](
/// https://gpuweb.github.io/gpuweb/#dictdef-gpushadermoduledescriptor).
#[derive(Clone, Debug)]
pub struct ShaderModuleDescriptor<'a> {
    /// Debug label of the shader module. This will show up in graphics debuggers for easy identification.
    pub label: Label<'a>,
    /// Source code for the shader.
    pub source: ShaderSource<'a>,
}
static_assertions::assert_impl_all!(ShaderModuleDescriptor<'_>: Send, Sync);

/// Descriptor for a shader module given by any of several sources.
/// At least one of the shader types that may be used by the backend must be `Some`
///
/// This type is unique to the Rust API of `wgpu`. In the WebGPU specification,
/// only WGSL source code strings are accepted.
pub type ShaderModuleDescriptorPassthrough<'a> =
    wgt::CreateShaderModuleDescriptorPassthrough<'a, Label<'a>>;

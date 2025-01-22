//! Types for dealing with Instances

use alloc::string::String;

use crate::Backends;

/// Options for creating an instance.
#[derive(Clone, Debug)]
pub struct InstanceDescriptor {
    /// Which `Backends` to enable.
    pub backends: Backends,
    /// Flags to tune the behavior of the instance.
    pub flags: InstanceFlags,
    /// Options the control the behavior of various backends.
    pub backend_options: BackendOptions,
}

impl Default for InstanceDescriptor {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            flags: InstanceFlags::default(),
            backend_options: BackendOptions::default(),
        }
    }
}

impl InstanceDescriptor {
    /// Choose instance options entirely from environment variables.
    ///
    /// This is equivalent to calling `from_env` on every field.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        Self::default().with_env()
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let backends = self.backends.with_env();
        let flags = self.flags.with_env();
        let backend_options = self.backend_options.with_env();
        Self {
            backends,
            flags,
            backend_options,
        }
    }
}

bitflags::bitflags! {
    /// Instance debugging flags.
    ///
    /// These are not part of the webgpu standard.
    ///
    /// Defaults to enabling debugging-related flags if the build configuration has `debug_assertions`.
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct InstanceFlags: u32 {
        /// Generate debug information in shaders and objects.
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_DEBUG` environment variable.
        const DEBUG = 1 << 0;
        /// Enable validation, if possible.
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_VALIDATION` environment variable.
        const VALIDATION = 1 << 1;
        /// Don't pass labels to wgpu-hal.
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_DISCARD_HAL_LABELS` environment variable.
        const DISCARD_HAL_LABELS = 1 << 2;
        /// Whether wgpu should expose adapters that run on top of non-compliant adapters.
        ///
        /// Turning this on might mean that some of the functionality provided by the wgpu
        /// adapter/device is not working or is broken. It could be that all the functionality
        /// wgpu currently exposes works but we can't tell for sure since we have no additional
        /// transparency into what is working and what is not on the underlying adapter.
        ///
        /// This mainly applies to a Vulkan driver's compliance version. If the major compliance version
        /// is `0`, then the driver is ignored. This flag allows that driver to be enabled for testing.
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER` environment variable.
        const ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER = 1 << 3;
        /// Enable GPU-based validation. Implies [`Self::VALIDATION`]. Currently, this only changes
        /// behavior on the DX12 and Vulkan backends.
        ///
        /// Supported platforms:
        ///
        /// - D3D12; called ["GPU-based validation", or
        ///   "GBV"](https://web.archive.org/web/20230206120404/https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-d3d12-debug-layer-gpu-based-validation)
        /// - Vulkan, via the `VK_LAYER_KHRONOS_validation` layer; called ["GPU-Assisted
        ///   Validation"](https://github.com/KhronosGroup/Vulkan-ValidationLayers/blob/e45aeb85079e0835694cb8f03e6681fd18ae72c9/docs/gpu_validation.md#gpu-assisted-validation)
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_GPU_BASED_VALIDATION` environment variable.
        const GPU_BASED_VALIDATION = 1 << 4;
    }
}

impl Default for InstanceFlags {
    fn default() -> Self {
        Self::from_build_config()
    }
}

impl InstanceFlags {
    /// Enable recommended debugging and validation flags.
    #[must_use]
    pub fn debugging() -> Self {
        InstanceFlags::DEBUG | InstanceFlags::VALIDATION
    }

    /// Enable advanced debugging and validation flags (potentially very slow).
    #[must_use]
    pub fn advanced_debugging() -> Self {
        Self::debugging() | InstanceFlags::GPU_BASED_VALIDATION
    }

    /// Infer decent defaults from the build type.
    ///
    /// If cfg!(debug_assertions) is true, then this returns [`Self::debugging()`].
    /// Otherwise, it returns [`Self::empty()`].
    #[must_use]
    pub fn from_build_config() -> Self {
        if cfg!(debug_assertions) {
            return InstanceFlags::debugging();
        }

        InstanceFlags::empty()
    }

    /// Derive defaults from environment variables. See [`Self::with_env()`] for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        Self::default().with_env()
    }

    /// Takes the given flags, modifies them based on the environment variables, and returns the result.
    ///
    /// - If an environment variable is set to anything but "0", the corresponding flag is set.
    /// - If the value is "0", the flag is unset.
    /// - If the environment variable is not present, then the flag retains its initial value.
    ///
    /// For example `let flags = InstanceFlags::debugging().with_env();` with `WGPU_VALIDATION=0`
    /// does not contain `InstanceFlags::VALIDATION`.
    ///
    /// The environment variables are named after the flags prefixed with "WGPU_". For example:
    /// - `WGPU_DEBUG`
    /// - `WGPU_VALIDATION`
    /// - `WGPU_DISCARD_HAL_LABELS`
    /// - `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER`
    /// - `WGPU_GPU_BASED_VALIDATION`
    #[must_use]
    pub fn with_env(mut self) -> Self {
        fn env(key: &str) -> Option<bool> {
            crate::env::var(key).map(|s| match s.as_str() {
                "0" => false,
                _ => true,
            })
        }

        if let Some(bit) = env("WGPU_VALIDATION") {
            self.set(Self::VALIDATION, bit);
        }
        if let Some(bit) = env("WGPU_DEBUG") {
            self.set(Self::DEBUG, bit);
        }
        if let Some(bit) = env("WGPU_DISCARD_HAL_LABELS") {
            self.set(Self::DISCARD_HAL_LABELS, bit);
        }
        if let Some(bit) = env("WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER") {
            self.set(Self::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER, bit);
        }
        if let Some(bit) = env("WGPU_GPU_BASED_VALIDATION") {
            self.set(Self::GPU_BASED_VALIDATION, bit);
        }

        self
    }
}

/// Options that are passed to a given backend.
#[derive(Clone, Debug, Default)]
pub struct BackendOptions {
    /// Options for the OpenGL/OpenGLES backend.
    pub gl: GlBackendOptions,
    /// Options for the DX12 backend.
    pub dx12: Dx12BackendOptions,
}

impl BackendOptions {
    /// Choose backend options by calling `from_env` on every field.
    ///
    /// See those methods for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let gl = GlBackendOptions::from_env_or_default();
        let dx12 = Dx12BackendOptions::from_env_or_default();
        Self { gl, dx12 }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let gl = self.gl.with_env();
        let dx12 = self.dx12.with_env();
        Self { gl, dx12 }
    }
}

/// Configuration for the OpenGL/OpenGLES backend.
#[derive(Clone, Debug, Default)]
pub struct GlBackendOptions {
    /// Which OpenGL ES 3 minor version to request, if using OpenGL ES.
    pub gles_minor_version: Gles3MinorVersion,
    /// Behavior of OpenGL fences. Affects how `on_completed_work_done` and `device.poll` behave.
    pub short_circuit_fences: GlFenceBehavior,
}

impl GlBackendOptions {
    /// Choose OpenGL backend options by calling `from_env` on every field.
    ///
    /// See those methods for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let gles_minor_version = Gles3MinorVersion::from_env().unwrap_or_default();
        Self {
            gles_minor_version,
            short_circuit_fences: GlFenceBehavior::Normal,
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let gles_minor_version = self.gles_minor_version.with_env();
        let short_circuit_fences = self.short_circuit_fences.with_env();
        Self {
            gles_minor_version,
            short_circuit_fences,
        }
    }
}

/// Configuration for the DX12 backend.
#[derive(Clone, Debug, Default)]
pub struct Dx12BackendOptions {
    /// Which DX12 shader compiler to use.
    pub shader_compiler: Dx12Compiler,
}

impl Dx12BackendOptions {
    /// Choose DX12 backend options by calling `from_env` on every field.
    ///
    /// See those methods for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let compiler = Dx12Compiler::from_env().unwrap_or_default();
        Self {
            shader_compiler: compiler,
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let shader_compiler = self.shader_compiler.with_env();
        Self { shader_compiler }
    }
}

/// Selects which DX12 shader compiler to use.
///
/// If the `DynamicDxc` option is selected, but `dxcompiler.dll` and `dxil.dll` files aren't found,
/// then this will fall back to the Fxc compiler at runtime and log an error.
#[derive(Clone, Debug, Default)]
pub enum Dx12Compiler {
    /// The Fxc compiler (default) is old, slow and unmaintained.
    ///
    /// However, it doesn't require any additional .dlls to be shipped with the application.
    #[default]
    Fxc,
    /// The Dxc compiler is new, fast and maintained.
    ///
    /// However, it requires both `dxcompiler.dll` and `dxil.dll` to be shipped with the application.
    /// These files can be downloaded from <https://github.com/microsoft/DirectXShaderCompiler/releases>.
    ///
    /// Minimum supported version: [v1.5.2010](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.5.2010)
    ///
    /// It also requires WDDM 2.1 (Windows 10 version 1607).
    DynamicDxc {
        /// Path to `dxcompiler.dll`.
        dxc_path: String,
        /// Path to `dxil.dll`.
        dxil_path: String,
    },
    /// The statically-linked variant of Dxc.
    ///
    /// The `static-dxc` feature is required for this setting to be used successfully on DX12.
    /// Not available on `windows-aarch64-pc-*` targets.
    StaticDxc,
}

impl Dx12Compiler {
    /// Helper function to construct a `DynamicDxc` variant with default paths.
    pub fn default_dynamic_dxc() -> Self {
        Self::DynamicDxc {
            dxc_path: String::from("dxcompiler.dll"),
            dxil_path: String::from("dxil.dll"),
        }
    }

    /// Choose which DX12 shader compiler to use from the environment variable `WGPU_DX12_COMPILER`.
    ///
    /// Valid values, case insensitive:
    /// - `Fxc`
    /// - `Dxc` or `DynamicDxc`
    /// - `StaticDxc`
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let value = crate::env::var("WGPU_DX12_COMPILER")
            .as_deref()?
            .to_lowercase();
        match value.as_str() {
            "dxc" | "dynamicdxc" => Some(Self::default_dynamic_dxc()),
            "staticdxc" => Some(Self::StaticDxc),
            "fxc" => Some(Self::Fxc),
            _ => None,
        }
    }

    /// Takes the given compiler, modifies it based on the `WGPU_DX12_COMPILER` environment variable, and returns the result.
    ///
    /// See `from_env` for more information.
    #[must_use]
    pub fn with_env(self) -> Self {
        if let Some(compiler) = Self::from_env() {
            compiler
        } else {
            self
        }
    }
}

/// Selects which OpenGL ES 3 minor version to request.
///
/// When using ANGLE as an OpenGL ES/EGL implementation, explicitly requesting `Version1` can provide a non-conformant ES 3.1 on APIs like D3D11.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub enum Gles3MinorVersion {
    /// No explicit minor version is requested, the driver automatically picks the highest available.
    #[default]
    Automatic,

    /// Request an ES 3.0 context.
    Version0,

    /// Request an ES 3.1 context.
    Version1,

    /// Request an ES 3.2 context.
    Version2,
}

impl Gles3MinorVersion {
    /// Choose which minor OpenGL ES version to use from the environment variable `WGPU_GLES_MINOR_VERSION`.
    ///
    /// Possible values are `0`, `1`, `2` or `automatic`. Case insensitive.
    ///
    /// Use with `unwrap_or_default()` to get the default value if the environment variable is not set.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let value = crate::env::var("WGPU_GLES_MINOR_VERSION")
            .as_deref()?
            .to_lowercase();
        match value.as_str() {
            "automatic" => Some(Self::Automatic),
            "0" => Some(Self::Version0),
            "1" => Some(Self::Version1),
            "2" => Some(Self::Version2),
            _ => None,
        }
    }

    /// Takes the given compiler, modifies it based on the `WGPU_GLES_MINOR_VERSION` environment variable, and returns the result.
    ///
    /// See `from_env` for more information.
    #[must_use]
    pub fn with_env(self) -> Self {
        if let Some(compiler) = Self::from_env() {
            compiler
        } else {
            self
        }
    }
}

/// Dictate the behavior of fences in OpenGL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GlFenceBehavior {
    /// Fences in OpenGL behave normally. If you don't know what to pick, this is what you want.
    #[default]
    Normal,
    /// Fences in OpenGL are short-circuited to always return `true` immediately.
    ///
    /// This solves a very specific issue that arose due to a bug in wgpu-core that made
    /// many WebGL programs work when they "shouldn't" have. If you have code that is trying
    /// to call `device.poll(wgpu::Maintain::Wait)` on WebGL, you need to enable this option
    /// for the "Wait" to behave how you would expect.
    ///
    /// Previously all `poll(Wait)` acted like the OpenGL fences were signalled even if they weren't.
    /// See <https://github.com/gfx-rs/wgpu/issues/4589> for more information.
    ///
    /// When this is set `Queue::on_completed_work_done` will always return the next time the device
    /// is maintained, not when the work is actually done on the GPU.
    AutoFinish,
}

impl GlFenceBehavior {
    /// Returns true if the fence behavior is `AutoFinish`.
    pub fn is_auto_finish(&self) -> bool {
        matches!(self, Self::AutoFinish)
    }

    /// Returns true if the fence behavior is `Normal`.
    pub fn is_normal(&self) -> bool {
        matches!(self, Self::Normal)
    }

    /// Choose which minor OpenGL ES version to use from the environment variable `WGPU_GL_FENCE_BEHAVIOR`.
    ///
    /// Possible values are `Normal` or `AutoFinish`. Case insensitive.
    ///
    /// Use with `unwrap_or_default()` to get the default value if the environment variable is not set.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let value = crate::env::var("WGPU_GL_FENCE_BEHAVIOR")
            .as_deref()?
            .to_lowercase();
        match value.as_str() {
            "normal" => Some(Self::Normal),
            "autofinish" => Some(Self::AutoFinish),
            _ => None,
        }
    }

    /// Takes the given compiler, modifies it based on the `WGPU_GL_FENCE_BEHAVIOR` environment variable, and returns the result.
    ///
    /// See `from_env` for more information.
    #[must_use]
    pub fn with_env(self) -> Self {
        if let Some(fence) = Self::from_env() {
            fence
        } else {
            self
        }
    }
}

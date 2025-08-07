//! Types for dealing with Instances

use alloc::string::String;

use crate::Backends;

#[cfg(doc)]
use crate::{Backend, DownlevelFlags};

/// Options for creating an instance.
///
/// If you want to allow control of instance settings via environment variables, call either
/// [`InstanceDescriptor::from_env_or_default()`] or [`InstanceDescriptor::with_env()`]. Each type
/// within this descriptor has its own equivalent methods, so you can select which options you want
/// to expose to influence from the environment.
#[derive(Clone, Debug, Default)]
pub struct InstanceDescriptor {
    /// Which [`Backends`] to enable.
    ///
    /// [`Backends::BROWSER_WEBGPU`] has an additional effect:
    /// If it is set and a [`navigator.gpu`](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/gpu)
    /// object is present, this instance will *only* be able to create WebGPU adapters.
    ///
    /// ⚠️ On some browsers this check is insufficient to determine whether WebGPU is supported,
    /// as the browser may define the `navigator.gpu` object, but be unable to create any WebGPU adapters.
    /// For targeting _both_ WebGPU & WebGL, it is recommended to use [`crate::util::new_instance_with_webgpu_detection`](../wgpu/util/fn.new_instance_with_webgpu_detection.html).
    ///
    /// If you instead want to force use of WebGL, either disable the `webgpu` compile-time feature
    /// or don't include the [`Backends::BROWSER_WEBGPU`] flag in this field.
    /// If it is set and WebGPU support is *not* detected, the instance will use `wgpu-core`
    /// to create adapters, meaning that if the `webgl` feature is enabled, it is able to create
    /// a WebGL adapter.
    pub backends: Backends,
    /// Flags to tune the behavior of the instance.
    pub flags: InstanceFlags,
    /// Memory budget thresholds used by some backends.
    pub memory_budget_thresholds: MemoryBudgetThresholds,
    /// Options the control the behavior of specific backends.
    pub backend_options: BackendOptions,
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
            memory_budget_thresholds: MemoryBudgetThresholds::default(),
            backend_options,
        }
    }
}

bitflags::bitflags! {
    /// Instance debugging flags.
    ///
    /// These are not part of the WebGPU standard.
    ///
    /// Defaults to enabling debugging-related flags if the build configuration has `debug_assertions`.
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct InstanceFlags: u32 {
        /// Generate debug information in shaders and objects.
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_DEBUG` environment variable.
        const DEBUG = 1 << 0;
        /// Enable validation in the backend API, if possible:
        ///
        /// - On the Direct3D `dx12` backend, this calls [`ID3D12Debug::EnableDebugLayer`][dx12].
        ///
        /// - On the Vulkan backend, this enables the [Vulkan Validation Layer][vvl].
        ///
        /// - On the `gles` backend driving Windows OpenGL, this enables [debug
        ///   output][gl:do], effectively calling `glEnable(GL_DEBUG_OUTPUT)`.
        ///
        /// - On non-Windows `gles` backends, this calls
        ///   [`eglDebugMessageControlKHR`][gl:dm] to enable all debugging messages.
        ///   If the GLES implementation is ANGLE running on Vulkan, this also
        ///   enables the Vulkan validation layers by setting
        ///   [`EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED`][gl:av].
        ///
        /// When `Self::from_env()` is used, this bit is set if the `WGPU_VALIDATION`
        /// environment variable has any value but "0".
        ///
        /// [dx12]: https://learn.microsoft.com/en-us/windows/win32/api/d3d12sdklayers/nf-d3d12sdklayers-id3d12debug-enabledebuglayer
        /// [vvl]: https://github.com/KhronosGroup/Vulkan-ValidationLayers
        /// [gl:dm]: https://registry.khronos.org/EGL/extensions/KHR/EGL_KHR_debug.txt
        /// [gl:do]: https://www.khronos.org/opengl/wiki/Debug_Output
        /// [gl:av]: https://chromium.googlesource.com/angle/angle/+/HEAD/extensions/EGL_ANGLE_platform_angle.txt
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

        /// Validate indirect buffer content prior to issuing indirect draws/dispatches.
        ///
        /// This validation will transform indirect calls into no-ops if they are not valid:
        ///
        /// - When calling `dispatch_workgroups_indirect`, all 3 indirect arguments encoded in the buffer
        /// must be less than the `max_compute_workgroups_per_dimension` device limit.
        /// - When calling `draw_indirect`/`draw_indexed_indirect`/`multi_draw_indirect`/`multi_draw_indexed_indirect`:
        ///   - If `Features::INDIRECT_FIRST_INSTANCE` is not enabled on the device, the `first_instance` indirect argument must be 0.
        ///   - The `first_instance` & `instance_count` indirect arguments must form a range that fits within all bound vertex buffers with `step_mode` set to `Instance`.
        /// - When calling `draw_indirect`/`multi_draw_indirect`:
        ///   - The `first_vertex` & `vertex_count` indirect arguments must form a range that fits within all bound vertex buffers with `step_mode` set to `Vertex`.
        /// - When calling `draw_indexed_indirect`/`multi_draw_indexed_indirect`:
        ///   - The `first_index` & `index_count` indirect arguments must form a range that fits within the bound index buffer.
        ///
        /// __Behavior is undefined if this validation is disabled and the rules above are not satisfied.__
        ///
        /// Disabling this will also cause the following built-ins to not report the right values on the D3D12 backend:
        ///
        /// - the 3 components of `@builtin(num_workgroups)` will be 0
        /// - the value of `@builtin(vertex_index)` will not take into account the value of the `first_vertex`/`base_vertex` argument present in the indirect buffer
        /// - the value of `@builtin(instance_index)` will not take into account the value of the `first_instance` argument present in the indirect buffer
        ///
        /// When `Self::from_env()` is used takes value from `WGPU_VALIDATION_INDIRECT_CALL` environment variable.
        const VALIDATION_INDIRECT_CALL = 1 << 5;

        /// Enable automatic timestamp normalization. This means that in [`CommandEncoder::resolve_query_set`][rqs],
        /// the timestamps will automatically be normalized to be in nanoseconds instead of the raw timestamp values.
        ///
        /// This is disabled by default because it introduces a compute shader into the resolution of query sets.
        ///
        /// This can be useful for users that need to read timestamps on the gpu, as the normalization
        /// can be a hassle to do manually. When this is enabled, the timestamp period returned by the queue
        /// will always be `1.0`.
        ///
        /// [rqs]: ../wgpu/struct.CommandEncoder.html#method.resolve_query_set
        const AUTOMATIC_TIMESTAMP_NORMALIZATION = 1 << 6;
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
        InstanceFlags::DEBUG | InstanceFlags::VALIDATION | InstanceFlags::VALIDATION_INDIRECT_CALL
    }

    /// Enable advanced debugging and validation flags (potentially very slow).
    #[must_use]
    pub fn advanced_debugging() -> Self {
        Self::debugging() | InstanceFlags::GPU_BASED_VALIDATION
    }

    /// Infer decent defaults from the build type.
    ///
    /// If `cfg!(debug_assertions)` is true, then this returns [`Self::debugging()`].
    /// Otherwise, it returns [`Self::empty()`].
    #[must_use]
    pub fn from_build_config() -> Self {
        if cfg!(debug_assertions) {
            return InstanceFlags::debugging();
        }

        InstanceFlags::VALIDATION_INDIRECT_CALL
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
    /// does not contain [`InstanceFlags::VALIDATION`].
    ///
    /// The environment variables are named after the flags prefixed with "WGPU_". For example:
    /// - `WGPU_DEBUG`
    /// - `WGPU_VALIDATION`
    /// - `WGPU_DISCARD_HAL_LABELS`
    /// - `WGPU_ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER`
    /// - `WGPU_GPU_BASED_VALIDATION`
    /// - `WGPU_VALIDATION_INDIRECT_CALL`
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
        if let Some(bit) = env("WGPU_VALIDATION_INDIRECT_CALL") {
            self.set(Self::VALIDATION_INDIRECT_CALL, bit);
        }

        self
    }
}

/// Memory budget thresholds used by backends to try to avoid high memory pressure situations.
///
/// Currently only the D3D12 and (optionally) Vulkan backends support these options.
#[derive(Default, Clone, Debug, Copy)]
pub struct MemoryBudgetThresholds {
    /// Threshold at which texture, buffer, query set and acceleration structure creation will start to return OOM errors.
    /// This is a percent of the memory budget reported by native APIs.
    ///
    /// If not specified, resource creation might still return OOM errors.
    pub for_resource_creation: Option<u8>,

    /// Threshold at which devices will become lost due to memory pressure.
    /// This is a percent of the memory budget reported by native APIs.
    ///
    /// If not specified, devices might still become lost due to memory pressure.
    pub for_device_loss: Option<u8>,
}

/// Options that are passed to a given backend.
///
/// Part of [`InstanceDescriptor`].
#[derive(Clone, Debug, Default)]
pub struct BackendOptions {
    /// Options for the OpenGL/OpenGLES backend, [`Backend::Gl`].
    pub gl: GlBackendOptions,
    /// Options for the DX12 backend, [`Backend::Dx12`].
    pub dx12: Dx12BackendOptions,
    /// Options for the noop backend, [`Backend::Noop`].
    pub noop: NoopBackendOptions,
}

impl BackendOptions {
    /// Choose backend options by calling `from_env` on every field.
    ///
    /// See those methods for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        Self {
            gl: GlBackendOptions::from_env_or_default(),
            dx12: Dx12BackendOptions::from_env_or_default(),
            noop: NoopBackendOptions::from_env_or_default(),
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        Self {
            gl: self.gl.with_env(),
            dx12: self.dx12.with_env(),
            noop: self.noop.with_env(),
        }
    }
}

/// Configuration for the OpenGL/OpenGLES backend.
///
/// Part of [`BackendOptions`].
#[derive(Clone, Debug, Default)]
pub struct GlBackendOptions {
    /// Which OpenGL ES 3 minor version to request, if using OpenGL ES.
    pub gles_minor_version: Gles3MinorVersion,
    /// Behavior of OpenGL fences. Affects how `on_completed_work_done` and `device.poll` behave.
    pub fence_behavior: GlFenceBehavior,
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
            fence_behavior: GlFenceBehavior::Normal,
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let gles_minor_version = self.gles_minor_version.with_env();
        let short_circuit_fences = self.fence_behavior.with_env();
        Self {
            gles_minor_version,
            fence_behavior: short_circuit_fences,
        }
    }
}

/// Configuration for the DX12 backend.
///
/// Part of [`BackendOptions`].
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

/// Configuration for the noop backend.
///
/// Part of [`BackendOptions`].
#[derive(Clone, Debug, Default)]
pub struct NoopBackendOptions {
    /// Whether to allow the noop backend to be used.
    ///
    /// The noop backend stubs out all operations except for buffer creation and mapping, so
    /// it must not be used when not expected. Therefore, it will not be used unless explicitly
    /// enabled.
    pub enable: bool,
}

impl NoopBackendOptions {
    /// Choose whether the noop backend is enabled from the environment.
    ///
    /// It will be enabled if the environment variable `WGPU_NOOP_BACKEND` has the value `1`
    /// and not otherwise. Future versions may assign other meanings to other values.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        Self {
            enable: Self::enable_from_env().unwrap_or(false),
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the
    /// result.
    ///
    /// See [`from_env_or_default()`](Self::from_env_or_default) for the interpretation.
    #[must_use]
    pub fn with_env(self) -> Self {
        Self {
            enable: Self::enable_from_env().unwrap_or(self.enable),
        }
    }

    fn enable_from_env() -> Option<bool> {
        let value = crate::env::var("WGPU_NOOP_BACKEND")?;
        match value.as_str() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        }
    }
}

/// DXC shader model.
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub enum DxcShaderModel {
    V6_0,
    V6_1,
    V6_2,
    V6_3,
    V6_4,
    V6_5,
    V6_6,
    V6_7,
}

/// Selects which DX12 shader compiler to use.
#[derive(Clone, Debug, Default)]
pub enum Dx12Compiler {
    /// The Fxc compiler (default) is old, slow and unmaintained.
    ///
    /// However, it doesn't require any additional .dlls to be shipped with the application.
    #[default]
    Fxc,
    /// The Dxc compiler is new, fast and maintained.
    ///
    /// However, it requires `dxcompiler.dll` to be shipped with the application.
    /// These files can be downloaded from <https://github.com/microsoft/DirectXShaderCompiler/releases>.
    ///
    /// Minimum supported version: [v1.8.2502](https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.8.2502)
    ///
    /// It also requires WDDM 2.1 (Windows 10 version 1607).
    DynamicDxc {
        /// Path to `dxcompiler.dll`.
        dxc_path: String,
        /// Maximum shader model the given dll supports.
        max_shader_model: DxcShaderModel,
    },
    /// The statically-linked variant of Dxc.
    ///
    /// The `static-dxc` feature is required for this setting to be used successfully on DX12.
    /// Not available on `windows-aarch64-pc-*` targets.
    StaticDxc,
}

impl Dx12Compiler {
    /// Helper function to construct a `DynamicDxc` variant with default paths.
    ///
    /// The dll must support at least shader model 6.8.
    pub fn default_dynamic_dxc() -> Self {
        Self::DynamicDxc {
            dxc_path: String::from("dxcompiler.dll"),
            max_shader_model: DxcShaderModel::V6_7, // should be 6.8 but the variant is missing
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
    /// to call `device.poll(wgpu::PollType::Wait)` on WebGL, you need to enable this option
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

//! [`Backend`], [`Backends`], and backend-specific options.

use alloc::string::String;
use core::hash::Hash;

#[cfg(any(feature = "serde", test))]
use serde::{Deserialize, Serialize};

use crate::link_to_wgpu_docs;

#[cfg(doc)]
use crate::InstanceDescriptor;

/// Backends supported by wgpu.
///
/// See also [`Backends`].
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Backend {
    /// Dummy backend, which may be used for testing.
    ///
    /// It performs no rendering or computation, but allows creation of stub GPU resource types,
    /// so that code which manages GPU resources can be tested without an available GPU.
    /// Specifically, the following operations are implemented:
    ///
    /// * Enumerating adapters will always return one noop adapter, which can be used to create
    ///   devices.
    /// * Buffers may be created, written, mapped, and copied to other buffers.
    /// * Command encoders may be created, but only buffer operations are useful.
    ///
    /// Other resources can be created but are nonfunctional; notably,
    ///
    /// * Render passes and compute passes are not executed.
    /// * Textures may be created, but do not store any texels.
    /// * There are no compatible surfaces.
    ///
    /// An adapter using the noop backend can only be obtained if [`NoopBackendOptions`]
    /// enables it, in addition to the ordinary requirement of [`Backends::NOOP`] being set.
    /// This ensures that applications not desiring a non-functional backend will not receive it.
    Noop = 0,
    /// Vulkan API (Windows, Linux, Android, MacOS via `vulkan-portability`/MoltenVK)
    Vulkan = 1,
    /// Metal API (Apple platforms)
    Metal = 2,
    /// Direct3D-12 (Windows)
    Dx12 = 3,
    /// OpenGL 3.3+ (Windows), OpenGL ES 3.0+ (Linux, Android, MacOS via Angle), and WebGL2
    Gl = 4,
    /// WebGPU in the browser
    BrowserWebGpu = 5,
}

impl Backend {
    /// Array of all [`Backend`] values, corresponding to [`Backends::all()`].
    pub const ALL: [Backend; Backends::all().bits().count_ones() as usize] = [
        Self::Noop,
        Self::Vulkan,
        Self::Metal,
        Self::Dx12,
        Self::Gl,
        Self::BrowserWebGpu,
    ];

    /// Returns the string name of the backend.
    #[must_use]
    pub const fn to_str(self) -> &'static str {
        match self {
            Backend::Noop => "noop",
            Backend::Vulkan => "vulkan",
            Backend::Metal => "metal",
            Backend::Dx12 => "dx12",
            Backend::Gl => "gl",
            Backend::BrowserWebGpu => "webgpu",
        }
    }
}

impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.to_str())
    }
}

bitflags::bitflags! {
    /// Represents the backends that wgpu will use.
    #[repr(transparent)]
    #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct Backends: u32 {
        /// [`Backend::Noop`].
        const NOOP = 1 << Backend::Noop as u32;

        /// [`Backend::Vulkan`].
        /// Supported on Windows, Linux/Android, and macOS/iOS via Vulkan Portability (with the Vulkan feature enabled)
        const VULKAN = 1 << Backend::Vulkan as u32;

        /// [`Backend::Gl`].
        /// Supported on Linux/Android, the web through webassembly via WebGL, and Windows and
        /// macOS/iOS via ANGLE
        const GL = 1 << Backend::Gl as u32;

        /// [`Backend::Metal`].
        /// Supported on macOS and iOS.
        const METAL = 1 << Backend::Metal as u32;

        /// [`Backend::Dx12`].
        /// Supported on Windows 10 and later
        const DX12 = 1 << Backend::Dx12 as u32;

        /// [`Backend::BrowserWebGpu`].
        /// Supported when targeting the web through WebAssembly with the `webgpu` feature enabled.
        ///
        /// The WebGPU backend is special in several ways:
        /// It is not not implemented by `wgpu_core` and instead by the higher level `wgpu` crate.
        /// Whether WebGPU is targeted is decided upon the creation of the `wgpu::Instance`,
        /// *not* upon adapter creation. See `wgpu::Instance::new`.
        const BROWSER_WEBGPU = 1 << Backend::BrowserWebGpu as u32;

        /// All the apis that wgpu offers first tier of support for.
        ///
        /// * [`Backends::VULKAN`]
        /// * [`Backends::METAL`]
        /// * [`Backends::DX12`]
        /// * [`Backends::BROWSER_WEBGPU`]
        const PRIMARY = Self::VULKAN.bits()
            | Self::METAL.bits()
            | Self::DX12.bits()
            | Self::BROWSER_WEBGPU.bits();

        /// All the apis that wgpu offers second tier of support for. These may
        /// be unsupported/still experimental.
        ///
        /// * [`Backends::GL`]
        const SECONDARY = Self::GL.bits();
    }
}

impl Default for Backends {
    fn default() -> Self {
        Self::all()
    }
}

impl From<Backend> for Backends {
    fn from(backend: Backend) -> Self {
        Self::from_bits(1 << backend as u32).unwrap()
    }
}

impl Backends {
    /// Gets a set of backends from the environment variable `WGPU_BACKEND`.
    ///
    /// See [`Self::from_comma_list()`] for the format of the string.
    pub fn from_env() -> Option<Self> {
        let env = crate::env::var("WGPU_BACKEND")?;
        Some(Self::from_comma_list(&env))
    }

    /// Takes the given options, modifies them based on the `WGPU_BACKEND` environment variable, and returns the result.
    pub fn with_env(&self) -> Self {
        if let Some(env) = Self::from_env() {
            env
        } else {
            *self
        }
    }

    /// Generates a set of backends from a comma separated list of case-insensitive backend names.
    ///
    /// Whitespace is stripped, so both 'gl, dx12' and 'gl,dx12' are valid.
    ///
    /// Always returns WEBGPU on wasm over webgpu.
    ///
    /// Names:
    /// - vulkan = "vulkan" or "vk"
    /// - dx12   = "dx12" or "d3d12"
    /// - metal  = "metal" or "mtl"
    /// - gles   = "opengl" or "gles" or "gl"
    /// - webgpu = "webgpu"
    pub fn from_comma_list(string: &str) -> Self {
        let mut backends = Self::empty();
        for backend in string.to_lowercase().split(',') {
            backends |= match backend.trim() {
                "vulkan" | "vk" => Self::VULKAN,
                "dx12" | "d3d12" => Self::DX12,
                "metal" | "mtl" => Self::METAL,
                "opengl" | "gles" | "gl" => Self::GL,
                "webgpu" => Self::BROWSER_WEBGPU,
                "noop" => Self::NOOP,
                b => {
                    log::warn!("unknown backend string '{b}'");
                    continue;
                }
            }
        }

        if backends.is_empty() {
            log::warn!("no valid backend strings found!");
        }

        backends
    }
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
    /// Controls whether debug functions (`glPushDebugGroup`, `glPopDebugGroup`,
    /// `glObjectLabel`, etc.) are enabled when supported by the driver.
    ///
    /// By default ([`GlDebugFns::Auto`]), debug functions are automatically
    /// disabled on devices with known bugs (e.g., Mali GPUs can crash in
    /// `glPushDebugGroup`). Use [`GlDebugFns::ForceEnabled`] to override this
    /// behavior, or [`GlDebugFns::Disabled`] to disable debug functions entirely.
    ///
    /// See also [`InstanceFlags::DISCARD_HAL_LABELS`], which prevents debug
    /// markers and labels from being sent to *any* backend, but without the
    /// driver-specific bug workarounds provided here.
    ///
    /// [`InstanceFlags::DISCARD_HAL_LABELS`]: crate::InstanceFlags::DISCARD_HAL_LABELS
    pub debug_fns: GlDebugFns,
}

impl GlBackendOptions {
    /// Choose OpenGL backend options by calling `from_env` on every field.
    ///
    /// See those methods for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let gles_minor_version = Gles3MinorVersion::from_env().unwrap_or_default();
        let debug_fns = GlDebugFns::from_env().unwrap_or_default();
        Self {
            gles_minor_version,
            fence_behavior: GlFenceBehavior::Normal,
            debug_fns,
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let gles_minor_version = self.gles_minor_version.with_env();
        let fence_behavior = self.fence_behavior.with_env();
        let debug_fns = self.debug_fns.with_env();
        Self {
            gles_minor_version,
            fence_behavior,
            debug_fns,
        }
    }
}

/// Controls whether OpenGL debug functions are enabled.
///
/// Debug functions include `glPushDebugGroup`, `glPopDebugGroup`, `glObjectLabel`, etc.
/// These are useful for debugging but can cause crashes on some buggy drivers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum GlDebugFns {
    /// Automatically decide whether to enable debug functions.
    ///
    /// Debug functions will be enabled if supported by the driver, unless
    /// running on a device known to have buggy debug function implementations
    /// (e.g., Mali GPUs which can crash in `glPushDebugGroup`).
    ///
    /// This is the default behavior.
    #[default]
    Auto,
    /// Force enable debug functions if supported by the driver.
    ///
    /// This ignores any device-specific workarounds and enables debug functions
    /// on all devices that support them, including those with known bugs.
    ForceEnabled,
    /// Disable debug functions entirely.
    ///
    /// Debug functions will not be used even if supported by the driver.
    Disabled,
}

impl GlDebugFns {
    /// Choose debug functions setting from the environment variable `WGPU_GL_DEBUG_FNS`.
    ///
    /// Possible values (case insensitive):
    /// - `auto` - automatically decide based on device
    /// - `forceenabled`, `force_enabled`, or `enabled` - force enable
    /// - `disabled` - disable entirely
    ///
    /// Use with `unwrap_or_default()` to get the default value if the environment variable is not set.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let value = crate::env::var("WGPU_GL_DEBUG_FNS")
            .as_deref()?
            .to_lowercase();
        match value.as_str() {
            "auto" => Some(Self::Auto),
            "forceenabled" | "force_enabled" | "enabled" => Some(Self::ForceEnabled),
            "disabled" => Some(Self::Disabled),
            _ => None,
        }
    }

    /// Takes the given setting, modifies it based on the `WGPU_GL_DEBUG_FNS` environment variable, and returns the result.
    ///
    /// See `from_env` for more information.
    #[must_use]
    pub fn with_env(self) -> Self {
        if let Some(debug_fns) = Self::from_env() {
            debug_fns
        } else {
            self
        }
    }
}

/// Used to force wgpu to expose certain features on passthrough shaders even when
/// those features aren't present on runtime-compiled shaders
#[derive(Default, Clone, Debug)]
pub struct ForceShaderModelToken {
    inner: Option<DxcShaderModel>,
}
impl ForceShaderModelToken {
    /// Creates an unsafe token, opting you in to seeing features that you may not necessarily use
    /// on standard runtime-compiled shaders.
    /// # Safety
    /// Do not make use in runtime-compiled shaders of any features that may not be supported by the FXC or DXC
    /// version you use.
    pub unsafe fn with_shader_model(sm: DxcShaderModel) -> Self {
        Self { inner: Some(sm) }
    }

    /// Returns the shader model version, if any, in this token.
    pub fn get(&self) -> Option<DxcShaderModel> {
        self.inner.clone()
    }
}

/// Configuration for the DX12 backend.
///
/// Part of [`BackendOptions`].
#[derive(Clone, Debug, Default)]
pub struct Dx12BackendOptions {
    /// Which DX12 shader compiler to use.
    pub shader_compiler: Dx12Compiler,
    /// Presentation system to use.
    pub presentation_system: Dx12SwapchainKind,
    /// Whether to wait for the latency waitable object before acquiring the next swapchain image.
    pub latency_waitable_object: Dx12UseFrameLatencyWaitableObject,
    /// For use with passthrough shaders. Expose features as if this shader model is present, even if you do not
    /// intend to ship DXC with your app.
    ///
    /// This does not override the device's shader model version, only the external shader compiler's version.
    pub force_shader_model: ForceShaderModelToken,
}

impl Dx12BackendOptions {
    /// Choose DX12 backend options by calling `from_env` on every field.
    ///
    /// See those methods for more information.
    #[must_use]
    pub fn from_env_or_default() -> Self {
        let compiler = Dx12Compiler::from_env().unwrap_or_default();
        let presentation_system = Dx12SwapchainKind::from_env().unwrap_or_default();
        let latency_waitable_object =
            Dx12UseFrameLatencyWaitableObject::from_env().unwrap_or_default();
        Self {
            shader_compiler: compiler,
            presentation_system,
            latency_waitable_object,
            force_shader_model: ForceShaderModelToken::default(),
        }
    }

    /// Takes the given options, modifies them based on the environment variables, and returns the result.
    ///
    /// This is equivalent to calling `with_env` on every field.
    #[must_use]
    pub fn with_env(self) -> Self {
        let shader_compiler = self.shader_compiler.with_env();
        let presentation_system = self.presentation_system.with_env();
        let latency_waitable_object = self.latency_waitable_object.with_env();
        Self {
            shader_compiler,
            presentation_system,
            latency_waitable_object,
            force_shader_model: ForceShaderModelToken::default(),
        }
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

#[derive(Clone, Debug, Default, Copy, PartialEq, Eq)]
/// Selects which kind of swapchain to use on DX12.
pub enum Dx12SwapchainKind {
    /// Use a DXGI swapchain made directly from the window's HWND.
    ///
    /// This does not support transparency but has better support from developer tooling from RenderDoc.
    #[default]
    DxgiFromHwnd,
    /// Use a DXGI swapchain made from a DirectComposition visual made automatically from the window's HWND.
    ///
    /// This creates a single [`IDCompositionVisual`] over the entire window that is used by the `Surface`.
    /// If a user wants to manage the composition tree themselves, they should create their own device and
    /// composition, and pass the relevant visual down via [`SurfaceTargetUnsafe::CompositionVisual`][CV].
    ///
    /// This supports transparent windows, but does not have support from RenderDoc.
    ///
    /// [`IDCompositionVisual`]: https://learn.microsoft.com/en-us/windows/win32/api/dcomp/nn-dcomp-idcompositionvisual
    #[doc = link_to_wgpu_docs!(["CV"]: "struct.SurfaceTargetUnsafe.html#variant.CompositionVisual")]
    DxgiFromVisual,
}

impl Dx12SwapchainKind {
    /// Choose which presentation system to use from the environment variable `WGPU_DX12_PRESENTATION_SYSTEM`.
    ///
    /// Valid values, case insensitive:
    /// - `DxgiFromVisual` or `Visual`
    /// - `DxgiFromHwnd` or `Hwnd` for [`Self::DxgiFromHwnd`]
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let value = crate::env::var("WGPU_DX12_PRESENTATION_SYSTEM")
            .as_deref()?
            .to_lowercase();
        match value.as_str() {
            "dxgifromvisual" | "visual" => Some(Self::DxgiFromVisual),
            "dxgifromhwnd" | "hwnd" => Some(Self::DxgiFromHwnd),
            _ => None,
        }
    }

    /// Takes the given presentation system, modifies it based on the `WGPU_DX12_PRESENTATION_SYSTEM` environment variable, and returns the result.
    ///
    /// See [`from_env`](Self::from_env) for more information.
    #[must_use]
    pub fn with_env(self) -> Self {
        if let Some(presentation_system) = Self::from_env() {
            presentation_system
        } else {
            self
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
    V6_8,
    V6_9,
}

impl DxcShaderModel {
    /// Get the shader model supported by a certain DXC version.
    pub fn from_dxc_version(major: u32, minor: u32) -> Self {
        // DXC version roughly has corresponded to shader model so far, where DXC 1.x supports SM 6.x.
        // See discussion in https://discord.com/channels/590611987420020747/996417435374714920/1471234702206701650.
        // Presumably DXC 2.0 and up will still support shader model 6.9.
        if major > 1 {
            Self::V6_9
        } else {
            Self::from_parts(6, minor)
        }
    }

    /// Parse a DxcShaderModel from its version components.
    pub fn from_parts(major: u32, minor: u32) -> Self {
        if major > 6 || minor > 8 {
            Self::V6_9
        } else {
            match minor {
                0 => DxcShaderModel::V6_0,
                1 => DxcShaderModel::V6_1,
                2 => DxcShaderModel::V6_2,
                3 => DxcShaderModel::V6_3,
                4 => DxcShaderModel::V6_4,
                5 => DxcShaderModel::V6_5,
                6 => DxcShaderModel::V6_6,
                7 => DxcShaderModel::V6_7,
                8 => DxcShaderModel::V6_8,
                9 => DxcShaderModel::V6_9,
                // > 6.9
                _ => DxcShaderModel::V6_9,
            }
        }
    }
}

/// Selects which DX12 shader compiler to use.
#[derive(Clone, Debug, Default)]
pub enum Dx12Compiler {
    /// The Fxc compiler (default) is old, slow and unmaintained.
    ///
    /// However, it doesn't require any additional .dlls to be shipped with the application.
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
    },
    /// The statically-linked variant of Dxc.
    ///
    /// The `static-dxc` feature is required for this setting to be used successfully on DX12.
    /// Not available on `windows-aarch64-pc-*` targets.
    StaticDxc,
    /// Use statically-linked DXC if available. Otherwise check for dynamically linked DXC on the PATH. Finally, fallback to FXC.
    #[default]
    Auto,
}

impl Dx12Compiler {
    /// Helper function to construct a `DynamicDxc` variant with default paths.
    ///
    /// The dll must support at least shader model 6.8.
    pub fn default_dynamic_dxc() -> Self {
        Self::DynamicDxc {
            dxc_path: String::from("dxcompiler.dll"),
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
            "auto" => Some(Self::Auto),
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

/// Whether and how to use a waitable handle obtained from `GetFrameLatencyWaitableObject`.
#[derive(Clone, Debug, Default)]
pub enum Dx12UseFrameLatencyWaitableObject {
    /// Do not obtain a waitable handle and do not wait for it. The swapchain will
    /// be created without the `DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT` flag.
    None,
    /// Obtain a waitable handle and wait for it before acquiring the next swapchain image.
    #[default]
    Wait,
    /// Create the swapchain with the `DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT` flag and
    /// obtain a waitable handle, but do not wait for it before acquiring the next swapchain image.
    /// This is useful if the application wants to wait for the waitable object itself.
    DontWait,
}

impl Dx12UseFrameLatencyWaitableObject {
    /// Choose whether to use a frame latency waitable object from the environment variable `WGPU_DX12_USE_FRAME_LATENCY_WAITABLE_OBJECT`.
    ///
    /// Valid values, case insensitive:
    /// - `None`
    /// - `Wait`
    /// - `DontWait`
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let value = crate::env::var("WGPU_DX12_USE_FRAME_LATENCY_WAITABLE_OBJECT")
            .as_deref()?
            .to_lowercase();
        match value.as_str() {
            "none" => Some(Self::None),
            "wait" => Some(Self::Wait),
            "dontwait" => Some(Self::DontWait),
            _ => None,
        }
    }

    /// Takes the given setting, modifies it based on the `WGPU_DX12_USE_FRAME_LATENCY_WAITABLE_OBJECT` environment variable, and returns the result.
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

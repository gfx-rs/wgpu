use crate::{
    backend,
    binding_model::MAX_BIND_GROUPS,
    device::BIND_BUFFER_ALIGNMENT,
    hub::{GfxBackend, Token, GLOBAL},
    id::{Input, Output},
    AdapterId,
    AdapterInfo,
    Backend,
    Device,
    DeviceId,
};
#[cfg(not(feature = "remote"))]
use crate::{gfx_select, SurfaceId};

#[cfg(not(feature = "remote"))]
use bitflags::bitflags;
use log::info;
#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

use hal::{
    self,
    Instance as _,
    adapter::PhysicalDevice as _,
    queue::QueueFamily as _,
};
#[cfg(not(feature = "remote"))]
use std::marker::PhantomData;


#[derive(Debug)]
pub struct Instance {
    #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
    vulkan: Option<gfx_backend_vulkan::Instance>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    metal: gfx_backend_metal::Instance,
    #[cfg(windows)]
    dx12: Option<gfx_backend_dx12::Instance>,
    #[cfg(windows)]
    dx11: gfx_backend_dx11::Instance,
}

impl Instance {
    pub(crate) fn new(name: &str, version: u32) -> Self {
        Instance {
            #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
            vulkan: gfx_backend_vulkan::Instance::create(name, version).ok(),
            #[cfg(any(target_os = "ios", target_os = "macos"))]
            metal: gfx_backend_metal::Instance::create(name, version).unwrap(),
            #[cfg(windows)]
            dx12: gfx_backend_dx12::Instance::create(name, version).ok(),
            #[cfg(windows)]
            dx11: gfx_backend_dx11::Instance::create(name, version).unwrap(),
        }
    }
}

type GfxSurface<B> = <B as hal::Backend>::Surface;

#[derive(Debug)]
pub struct Surface {
    #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
    pub(crate) vulkan: Option<GfxSurface<backend::Vulkan>>,
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    pub(crate) metal: GfxSurface<backend::Metal>,
    #[cfg(windows)]
    pub(crate) dx12: Option<GfxSurface<backend::Dx12>>,
    #[cfg(windows)]
    pub(crate) dx11: GfxSurface<backend::Dx11>,
}

#[derive(Debug)]
pub struct Adapter<B: hal::Backend> {
    pub(crate) raw: hal::adapter::Adapter<B>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum PowerPreference {
    Default = 0,
    LowPower = 1,
    HighPerformance = 2,
}

#[cfg(not(feature = "remote"))]
bitflags! {
    #[repr(transparent)]
    pub struct BackendBit: u32 {
        const VULKAN = 1 << Backend::Vulkan as u32;
        const GL = 1 << Backend::Gl as u32;
        const METAL = 1 << Backend::Metal as u32;
        const DX12 = 1 << Backend::Dx12 as u32;
        const DX11 = 1 << Backend::Dx11 as u32;
        const PRIMARY = Self::VULKAN.bits | Self::METAL.bits | Self::DX12.bits;
        const SECONDARY = Self::GL.bits | Self::DX11.bits;
    }
}

#[cfg(not(feature = "remote"))]
impl From<Backend> for BackendBit {
    fn from(backend: Backend) -> Self {
        BackendBit::from_bits(1 << backend as u32).unwrap()
    }
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct RequestAdapterOptions {
    pub power_preference: PowerPreference,
    #[cfg(not(feature = "remote"))]
    pub backends: BackendBit,
}

impl Default for RequestAdapterOptions {
    fn default() -> Self {
        RequestAdapterOptions {
            power_preference: PowerPreference::Default,
            #[cfg(not(feature = "remote"))]
            backends: BackendBit::PRIMARY,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct Extensions {
    pub anisotropic_filtering: bool,
}

#[repr(C)]
#[derive(Clone, Debug)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct Limits {
    pub max_bind_groups: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_bind_groups: MAX_BIND_GROUPS as u32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub struct DeviceDescriptor {
    pub extensions: Extensions,
    pub limits: Limits,
}

#[cfg(not(feature = "remote"))]
pub fn wgpu_create_surface(raw_handle: raw_window_handle::RawWindowHandle) -> SurfaceId {
    use raw_window_handle::RawWindowHandle as Rwh;

    let instance = &GLOBAL.instance;
    let surface = match raw_handle {
        #[cfg(target_os = "ios")]
        Rwh::IOS(h) => Surface {
            #[cfg(feature = "gfx-backend-vulkan")]
            vulkan: None,
            metal: instance
                .metal
                .create_surface_from_uiview(h.ui_view, cfg!(debug_assertions)),
        },
        #[cfg(target_os = "macos")]
        Rwh::MacOS(h) => Surface {
            #[cfg(feature = "gfx-backend-vulkan")]
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_ns_view(h.ns_view)),
            metal: instance
                .metal
                .create_surface_from_nsview(h.ns_view, cfg!(debug_assertions)),
        },
        #[cfg(all(unix, not(target_os = "ios"), not(target_os = "macos")))]
        Rwh::X11(h) => Surface {
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_xlib(h.display as _, h.window as _)),
        },
        #[cfg(all(unix, not(target_os = "ios"), not(target_os = "macos")))]
        Rwh::Wayland(h) => Surface {
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_wayland(h.display, h.surface)),
        },
        #[cfg(windows)]
        Rwh::Windows(h) => Surface {
            vulkan: instance
                .vulkan
                .as_ref()
                .map(|inst| inst.create_surface_from_hwnd(std::ptr::null_mut(), h.hwnd)),
            dx12: instance
                .dx12
                .as_ref()
                .map(|inst| inst.create_surface_from_hwnd(h.hwnd)),
            dx11: instance.dx11.create_surface_from_hwnd(h.hwnd),
        },
        _ => panic!("Unsupported window handle"),
    };

    let mut token = Token::root();
    GLOBAL
        .surfaces
        .register_identity(PhantomData, surface, &mut token)
}

#[cfg(all(not(feature = "remote"), unix, not(target_os = "ios"), not(target_os = "macos")))]
#[no_mangle]
pub extern "C" fn wgpu_create_surface_from_xlib(
    display: *mut *const std::ffi::c_void,
    window: u64,
) -> SurfaceId {
    use raw_window_handle::unix::X11Handle;
    wgpu_create_surface(raw_window_handle::RawWindowHandle::X11(X11Handle {
        window,
        display: display as *mut _,
        ..X11Handle::empty()
    }))
}

#[cfg(all(not(feature = "remote"), any(target_os = "ios", target_os = "macos")))]
#[no_mangle]
pub extern "C" fn wgpu_create_surface_from_metal_layer(layer: *mut std::ffi::c_void) -> SurfaceId {
    let surface = Surface {
        #[cfg(feature = "gfx-backend-vulkan")]
        vulkan: None, //TODO: currently requires `NSView`
        metal: GLOBAL
            .instance
            .metal
            .create_surface_from_layer(layer as *mut _, cfg!(debug_assertions)),
    };

    GLOBAL
        .surfaces
        .register_identity(PhantomData, surface, &mut Token::root())
}

#[cfg(all(not(feature = "remote"), windows))]
#[no_mangle]
pub extern "C" fn wgpu_create_surface_from_windows_hwnd(
    _hinstance: *mut std::ffi::c_void,
    hwnd: *mut std::ffi::c_void,
) -> SurfaceId {
    use raw_window_handle::windows::WindowsHandle;
    wgpu_create_surface(raw_window_handle::RawWindowHandle::Windows(
        raw_window_handle::windows::WindowsHandle {
            hwnd,
            ..WindowsHandle::empty()
        },
    ))
}

pub fn request_adapter(
    desc: &RequestAdapterOptions,
    input_ids: &[Input<AdapterId>],
) -> Option<Output<AdapterId>> {
    let instance = &GLOBAL.instance;
    let mut device_types = Vec::new();

    #[cfg(feature = "remote")]
    let find_input = |b: Backend| input_ids.iter().find(|id| id.backend() == b).cloned();
    #[cfg(not(feature = "remote"))]
    let find_input = |b: Backend| {
        let _ = input_ids;
        if desc.backends.contains(b.into()) {
            Some(PhantomData)
        } else {
            None
        }
    };

    let id_vulkan = find_input(Backend::Vulkan);
    let id_metal = find_input(Backend::Metal);
    let id_dx12 = find_input(Backend::Dx12);
    let id_dx11 = find_input(Backend::Dx11);

    #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
    let mut adapters_vk = match instance.vulkan {
        Some(ref inst) if id_vulkan.is_some() => {
            let adapters = inst.enumerate_adapters();
            device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
            adapters
        }
        _ => Vec::new(),
    };
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    let mut adapters_mtl = if id_metal.is_some() {
        let adapters = instance.metal.enumerate_adapters();
        device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
        adapters
    } else {
        Vec::new()
    };
    #[cfg(windows)]
    let mut adapters_dx12 = match instance.dx12 {
        Some(ref inst) if id_dx12.is_some() => {
            let adapters = inst.enumerate_adapters();
            device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
            adapters
        }
        _ => Vec::new(),
    };
    #[cfg(windows)]
    let mut adapters_dx11 = if id_dx11.is_some() {
        let adapters = instance.dx11.enumerate_adapters();
        device_types.extend(adapters.iter().map(|ad| ad.info.device_type.clone()));
        adapters
    } else {
        Vec::new()
    };

    if device_types.is_empty() {
        panic!("No adapters are available!");
    }

    let (mut integrated, mut discrete, mut virt, mut other) = (None, None, None, None);

    for (i, ty) in device_types.into_iter().enumerate() {
        match ty {
            hal::adapter::DeviceType::IntegratedGpu => {
                integrated = integrated.or(Some(i));
            }
            hal::adapter::DeviceType::DiscreteGpu => {
                discrete = discrete.or(Some(i));
            }
            hal::adapter::DeviceType::VirtualGpu => {
                virt = virt.or(Some(i));
            }
            _ => {
                other = other.or(Some(i));
            }
        }
    }

    let preferred_gpu = match desc.power_preference {
        PowerPreference::Default => integrated.or(discrete).or(other).or(virt),
        PowerPreference::LowPower => integrated.or(other).or(discrete).or(virt),
        PowerPreference::HighPerformance => discrete.or(other).or(integrated).or(virt),
    };
    let mut token = Token::root();

    let mut selected = preferred_gpu.unwrap_or(0);
    #[cfg(any(not(any(target_os = "ios", target_os = "macos")), feature = "gfx-backend-vulkan"))]
    {
        if selected < adapters_vk.len() {
            let adapter = Adapter {
                raw: adapters_vk.swap_remove(selected),
            };
            info!("Adapter Vulkan {:?}", adapter.raw.info);
            let id_out = backend::Vulkan::hub().adapters.register_identity(
                id_vulkan.unwrap(),
                adapter,
                &mut token,
            );
            return Some(id_out);
        }
        selected -= adapters_vk.len();
    }
    #[cfg(any(target_os = "ios", target_os = "macos"))]
    {
        if selected < adapters_mtl.len() {
            let adapter = Adapter {
                raw: adapters_mtl.swap_remove(selected),
            };
            info!("Adapter Metal {:?}", adapter.raw.info);
            let id_out = backend::Metal::hub().adapters.register_identity(
                id_metal.unwrap(),
                adapter,
                &mut token,
            );
            return Some(id_out);
        }
        selected -= adapters_mtl.len();
    }
    #[cfg(windows)]
    {
        if selected < adapters_dx12.len() {
            let adapter = Adapter {
                raw: adapters_dx12.swap_remove(selected),
            };
            info!("Adapter Dx12 {:?}", adapter.raw.info);
            let id_out = backend::Dx12::hub().adapters.register_identity(
                id_dx12.unwrap(),
                adapter,
                &mut token,
            );
            return Some(id_out);
        }
        selected -= adapters_dx12.len();
        if selected < adapters_dx11.len() {
            let adapter = Adapter {
                raw: adapters_dx11.swap_remove(selected),
            };
            info!("Adapter Dx11 {:?}", adapter.raw.info);
            let id_out = backend::Dx11::hub().adapters.register_identity(
                id_dx11.unwrap(),
                adapter,
                &mut token,
            );
            return Some(id_out);
        }
        selected -= adapters_dx11.len();
    }
    let _ = (selected, id_vulkan, id_metal, id_dx12, id_dx11);
    None
}

#[cfg(not(feature = "remote"))]
#[no_mangle]
pub extern "C" fn wgpu_request_adapter(desc: Option<&RequestAdapterOptions>) -> AdapterId {
    request_adapter(&desc.cloned().unwrap_or_default(), &[]).unwrap()
}

pub fn adapter_request_device<B: GfxBackend>(
    adapter_id: AdapterId,
    _desc: &DeviceDescriptor,
    id_in: Input<DeviceId>,
) -> Output<DeviceId> {
    let hub = B::hub();
    let mut token = Token::root();
    let device = {
        let (adapter_guard, _) = hub.adapters.read(&mut token);
        let adapter = &adapter_guard[adapter_id].raw;

        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                family.queue_type().supports_graphics()
            })
            .unwrap();
        let mut gpu = unsafe {
            adapter.physical_device
                .open(&[(family, &[1.0])], hal::Features::empty())
                .unwrap()
        };

        let limits = adapter.physical_device.limits();
        assert_eq!(
            0,
            BIND_BUFFER_ALIGNMENT % limits.min_storage_buffer_offset_alignment,
            "Adapter storage buffer offset alignment not compatible with WGPU"
        );
        assert_eq!(
            0,
            BIND_BUFFER_ALIGNMENT % limits.min_uniform_buffer_offset_alignment,
            "Adapter uniform buffer offset alignment not compatible with WGPU"
        );

        let mem_props = adapter.physical_device.memory_properties();

        let supports_texture_d24_s8 = adapter
            .physical_device
            .format_properties(Some(hal::format::Format::D24UnormS8Uint))
            .optimal_tiling
            .contains(hal::format::ImageFeature::DEPTH_STENCIL_ATTACHMENT);

        Device::new(
            gpu.device,
            adapter_id,
            gpu.queue_groups.swap_remove(0),
            mem_props,
            supports_texture_d24_s8,
        )
    };

    hub.devices.register_identity(id_in, device, &mut token)
}

#[cfg(not(feature = "remote"))]
#[no_mangle]
pub extern "C" fn wgpu_adapter_request_device(
    adapter_id: AdapterId,
    desc: Option<&DeviceDescriptor>,
) -> DeviceId {
    let desc = &desc.cloned().unwrap_or_default();
    gfx_select!(adapter_id => adapter_request_device(adapter_id, desc, PhantomData))
}

pub fn adapter_get_info<B: GfxBackend>(adapter_id: AdapterId) -> AdapterInfo {
    let hub = B::hub();
    let mut token = Token::root();
    let (adapter_guard, _) = hub.adapters.read(&mut token);
    let adapter = &adapter_guard[adapter_id];
    adapter.raw.info.clone()
}

#[cfg(not(feature = "remote"))]
pub fn wgpu_adapter_get_info(
    adapter_id: AdapterId
) -> AdapterInfo {
    gfx_select!(adapter_id => adapter_get_info(adapter_id))
}

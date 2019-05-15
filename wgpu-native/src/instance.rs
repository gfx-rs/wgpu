use crate::{
    binding_model::MAX_BIND_GROUPS,
    hub::HUB,
    AdapterHandle,
    AdapterId,
    DeviceHandle,
    InstanceId,
    SurfaceHandle,
};
#[cfg(feature = "local")]
use crate::{DeviceId, SurfaceId};

#[cfg(feature = "local")]
use log::info;
#[cfg(feature = "remote")]
use serde::{Deserialize, Serialize};

use hal::{self, Instance as _, PhysicalDevice as _};

#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
#[cfg_attr(feature = "remote", derive(Serialize, Deserialize))]
pub enum PowerPreference {
    Default = 0,
    LowPower = 1,
    HighPerformance = 2,
}

#[repr(C)]
#[derive(Debug)]
#[cfg_attr(feature = "remote", derive(Clone, Serialize, Deserialize))]
pub struct AdapterDescriptor {
    pub power_preference: PowerPreference,
}

#[repr(C)]
#[derive(Debug, Default)]
#[cfg_attr(feature = "remote", derive(Clone, Serialize, Deserialize))]
pub struct Extensions {
    pub anisotropic_filtering: bool,
}

#[repr(C)]
#[derive(Debug)]
#[cfg_attr(feature = "remote", derive(Clone, Serialize, Deserialize))]
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
#[derive(Debug)]
#[cfg_attr(feature = "remote", derive(Clone, Serialize, Deserialize))]
pub struct DeviceDescriptor {
    pub extensions: Extensions,
    pub limits: Limits,
}

pub fn create_instance() -> ::back::Instance {
    ::back::Instance::create("wgpu", 1)
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_create_instance() -> InstanceId {
    let inst = create_instance();
    HUB.instances.register_local(inst)
}

#[cfg(feature = "window-winit")]
#[no_mangle]
pub extern "C" fn wgpu_instance_create_surface_from_winit(
    instance_id: InstanceId,
    window: &winit::Window,
) -> SurfaceId {
    let raw = HUB.instances.read()[instance_id].create_surface(window);
    let surface = SurfaceHandle::new(raw);
    HUB.surfaces.register_local(surface)
}

#[allow(unused_variables)]
pub fn instance_create_surface_from_xlib(
    instance_id: InstanceId,
    display: *mut *const std::ffi::c_void,
    window: u64,
) -> SurfaceHandle {
    #[cfg(not(all(unix, feature = "gfx-backend-vulkan")))]
    unimplemented!();

    #[cfg(all(unix, feature = "gfx-backend-vulkan"))]
    SurfaceHandle::new(HUB.instances.read()[instance_id].create_surface_from_xlib(display, window))
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_instance_create_surface_from_xlib(
    instance_id: InstanceId,
    display: *mut *const std::ffi::c_void,
    window: u64,
) -> SurfaceId {
    let surface = instance_create_surface_from_xlib(instance_id, display, window);
    HUB.surfaces.register_local(surface)
}

#[allow(unused_variables)]
pub fn instance_create_surface_from_macos_layer(
    instance_id: InstanceId,
    layer: *mut std::ffi::c_void,
) -> SurfaceHandle {
    #[cfg(not(feature = "gfx-backend-metal"))]
    unimplemented!();

    #[cfg(feature = "gfx-backend-metal")]
    SurfaceHandle::new(
        HUB.instances.read()[instance_id]
            .create_surface_from_layer(layer as *mut _, cfg!(debug_assertions)),
    )
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_instance_create_surface_from_macos_layer(
    instance_id: InstanceId,
    layer: *mut std::ffi::c_void,
) -> SurfaceId {
    let surface = instance_create_surface_from_macos_layer(instance_id, layer);
    HUB.surfaces.register_local(surface)
}

#[allow(unused_variables)]
pub fn instance_create_surface_from_windows_hwnd(
    instance_id: InstanceId,
    hinstance: *mut std::ffi::c_void,
    hwnd: *mut std::ffi::c_void,
) -> SurfaceHandle {
    #[cfg(not(any(
        feature = "gfx-backend-dx11",
        feature = "gfx-backend-dx12",
        all(target_os = "windows", feature = "gfx-backend-vulkan"),
    )))]
    let raw = unimplemented!();

    #[cfg(any(feature = "gfx-backend-dx11", feature = "gfx-backend-dx12"))]
    let raw = HUB.instances.read()[instance_id].create_surface_from_hwnd(hwnd);

    #[cfg(all(target_os = "windows", feature = "gfx-backend-vulkan"))]
    let raw = HUB.instances.read()[instance_id].create_surface_from_hwnd(hinstance, hwnd);

    #[allow(unreachable_code)]
    SurfaceHandle::new(raw)
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_instance_create_surface_from_windows_hwnd(
    instance_id: InstanceId,
    hinstance: *mut std::ffi::c_void,
    hwnd: *mut std::ffi::c_void,
) -> SurfaceId {
    let surface = instance_create_surface_from_windows_hwnd(instance_id, hinstance, hwnd);
    HUB.surfaces.register_local(surface)
}

pub fn instance_get_adapter(instance_id: InstanceId, desc: &AdapterDescriptor) -> AdapterHandle {
    let instance_guard = HUB.instances.read();
    let instance = &instance_guard[instance_id];
    let (mut low, mut high, mut other) = (None, None, None);
    for adapter in instance.enumerate_adapters() {
        match adapter.info.device_type {
            hal::adapter::DeviceType::IntegratedGpu => low = Some(adapter),
            hal::adapter::DeviceType::DiscreteGpu => high = Some(adapter),
            _ => other = Some(adapter),
        }
    }

    let some = match desc.power_preference {
        PowerPreference::LowPower => low.or(high),
        PowerPreference::HighPerformance | PowerPreference::Default => high.or(low),
    };
    some
        .or(other)
        .expect("No adapters found. Please enable the feature for one of the graphics backends: vulkan, metal, dx12, dx11")
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_instance_get_adapter(
    instance_id: InstanceId,
    desc: &AdapterDescriptor,
) -> AdapterId {
    let adapter = instance_get_adapter(instance_id, desc);
    info!("Adapter {:?}", adapter.info);
    HUB.adapters.register_local(adapter)
}

pub fn adapter_create_device(adapter_id: AdapterId, _desc: &DeviceDescriptor) -> DeviceHandle {
    let adapter_guard = HUB.adapters.read();
    let adapter = &adapter_guard[adapter_id];
    let (raw, queue_group) = adapter.open_with::<_, hal::General>(1, |_qf| true).unwrap();
    let mem_props = adapter.physical_device.memory_properties();
    let limits = adapter.physical_device.limits();
    DeviceHandle::new(raw, adapter_id, queue_group, mem_props, limits)
}

#[cfg(feature = "local")]
#[no_mangle]
pub extern "C" fn wgpu_adapter_request_device(
    adapter_id: AdapterId,
    desc: &DeviceDescriptor,
) -> DeviceId {
    let device = adapter_create_device(adapter_id, desc);
    HUB.devices.register_local(device)
}

#[cfg(not(feature = "winit"))]
extern crate winit;

use hal::{self, Instance as _Instance, PhysicalDevice as _PhysicalDevice};

use crate::registry::{HUB, Items};
use crate::{WeaklyStored, Device, Surface,
    AdapterId, DeviceId, InstanceId, SurfaceId,
};


#[repr(C)]
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub enum PowerPreference {
    Default = 0,
    LowPower = 1,
    HighPerformance = 2,
}

#[repr(C)]
pub struct AdapterDescriptor {
    pub power_preference: PowerPreference,
}

#[repr(C)]
pub struct Extensions {
    pub anisotropic_filtering: bool,
}

#[repr(C)]
pub struct DeviceDescriptor {
    pub extensions: Extensions,
}

#[no_mangle]
pub extern "C" fn wgpu_create_instance() -> InstanceId {
    #[cfg(any(
        feature = "gfx-backend-vulkan",
        feature = "gfx-backend-dx12",
        feature = "gfx-backend-metal"
    ))]
    {
        let inst = ::back::Instance::create("wgpu", 1);
        HUB.instances.write().register(inst)
    }
    #[cfg(not(any(
        feature = "gfx-backend-vulkan",
        feature = "gfx-backend-dx12",
        feature = "gfx-backend-metal"
    )))]
    {
        unimplemented!()
    }
}

#[cfg(not(feature = "remote"))]
#[no_mangle]
pub extern "C" fn wgpu_instance_create_surface_from_winit(
    instance_id: InstanceId,
    window: &winit::Window,
) -> SurfaceId {
    let raw = HUB.instances
        .read()
        .get(instance_id)
        .create_surface(window);
    let surface = Surface {
        raw,
    };

    HUB.surfaces
        .write()
        .register(surface)
}

#[no_mangle]
pub extern "C" fn wgpu_instance_get_adapter(
    instance_id: InstanceId,
    desc: &AdapterDescriptor,
) -> AdapterId {
    let instance_guard = HUB.instances.read();
    let instance = instance_guard.get(instance_id);
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
    HUB.adapters.write().register(some.or(other).unwrap())
}

#[no_mangle]
pub extern "C" fn wgpu_adapter_create_device(
    adapter_id: AdapterId,
    _desc: &DeviceDescriptor,
) -> DeviceId {
    let mut adapter_guard = HUB.adapters.write();
    let adapter = adapter_guard.get_mut(adapter_id);
    let (raw, queue_group) = adapter.open_with::<_, hal::General>(1, |_qf| true).unwrap();
    let mem_props = adapter.physical_device.memory_properties();
    let device = Device::new(raw, WeaklyStored(adapter_id), queue_group, mem_props);

    HUB.devices
        .write()
        .register(device)
}

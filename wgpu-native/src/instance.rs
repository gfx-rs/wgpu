use hal::{self, Instance as _Instance, PhysicalDevice as _PhysicalDevice};

use registry;
use {AdapterId, Device, DeviceId, InstanceId};


#[repr(C)]
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
pub extern "C" fn create_instance() -> InstanceId {
    #[cfg(any(feature = "gfx-backend-vulkan", feature = "gfx-backend-dx12", feature = "gfx-backend-metal"))]
    {
        let mut registry = registry::INSTANCE_REGISTRY.lock().unwrap();
        let inst = ::back::Instance::create("wgpu", 1);
        registry.register(inst)
    }
    #[cfg(not(any(feature = "gfx-backend-vulkan", feature = "gfx-backend-dx12", feature = "gfx-backend-metal")))]
    {
        unimplemented!()
    }
}


#[no_mangle]
pub extern "C" fn instance_get_adapter(
    instance_id: InstanceId, desc: AdapterDescriptor
) -> AdapterId {
    let instance_registry = registry::INSTANCE_REGISTRY.lock().unwrap();
    let instance = instance_registry.get(instance_id).unwrap();
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
        PowerPreference::HighPerformance |
        PowerPreference::Default => high.or(low),
    };
    registry::ADAPTER_REGISTRY.lock().unwrap().register(some.or(other).unwrap())
}

#[no_mangle]
pub extern "C" fn adapter_create_device(
    adapter_id: AdapterId, desc: DeviceDescriptor
) -> DeviceId {
    let mut adapter_registry = registry::ADAPTER_REGISTRY.lock().unwrap();
    let adapter = adapter_registry.get_mut(adapter_id).unwrap();
    let (device, queue_group) = adapter.open_with::<_, hal::General>(1, |_qf| true).unwrap();
    let mem_props = adapter.physical_device.memory_properties();
    registry::DEVICE_REGISTRY.lock().unwrap().register(Device::new(device, queue_group, mem_props))
}

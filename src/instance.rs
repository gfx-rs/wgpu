use back;
use hal::{self, Instance as _Instance, PhysicalDevice as _PhysicalDevice};

use {AdapterHandle, Device, DeviceHandle, InstanceHandle};


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

pub extern "C"
fn create_instance() -> InstanceHandle {
    let inst = back::Instance::create("wgpu", 1);
    InstanceHandle::new(inst)
}

pub extern "C"
fn instance_get_adapter(
    instance: InstanceHandle, desc: AdapterDescriptor
) -> AdapterHandle {
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
    AdapterHandle::new(some.or(other).unwrap())
}

pub extern "C"
fn adapter_create_device(
    adapter: AdapterHandle, desc: DeviceDescriptor
) -> DeviceHandle {
    let queue_family = &adapter.queue_families[0];
    let gpu = adapter.physical_device.open(&[(queue_family, &[1f32])]).unwrap();
    DeviceHandle::new(Device {
        gpu,
        memory_properties: adapter.physical_device.memory_properties(),
    })
}

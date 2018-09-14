//use hal;

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
    anisotropic_filtering: bool,
}

#[repr(C)]
pub struct DeviceDescriptor {
    pub extension: Extensions,
}

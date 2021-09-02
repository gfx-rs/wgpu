#[cfg(feature = "renderdoc")]
pub(super) mod renderdoc;

pub mod db {
    pub mod intel {
        pub const VENDOR: u32 = 0x8086;
        pub const DEVICE_KABY_LAKE_MASK: u32 = 0x5900;
        pub const DEVICE_SKY_LAKE_MASK: u32 = 0x1900;
    }
    pub mod nvidia {
        pub const VENDOR: u32 = 0x10DE;
    }
}

pub fn map_naga_stage(stage: naga::ShaderStage) -> wgt::ShaderStages {
    match stage {
        naga::ShaderStage::Vertex => wgt::ShaderStages::VERTEX,
        naga::ShaderStage::Fragment => wgt::ShaderStages::FRAGMENT,
        naga::ShaderStage::Compute => wgt::ShaderStages::COMPUTE,
    }
}

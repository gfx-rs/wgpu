pub mod db {
    pub mod intel {
        pub const VENDOR: u32 = 0x8086;
        pub const DEVICE_KABY_LAKE_MASK: u32 = 0x5900;
        pub const DEVICE_SKY_LAKE_MASK: u32 = 0x1900;
    }
}

pub fn map_naga_stage(stage: naga::ShaderStage) -> wgt::ShaderStage {
    match stage {
        naga::ShaderStage::Vertex => wgt::ShaderStage::VERTEX,
        naga::ShaderStage::Fragment => wgt::ShaderStage::FRAGMENT,
        naga::ShaderStage::Compute => wgt::ShaderStage::COMPUTE,
    }
}
